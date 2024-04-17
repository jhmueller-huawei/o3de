/*
 * Copyright (c) Contributors to the Open 3D Engine Project.
 * For complete copyright and license terms please see the LICENSE at the root of this distribution.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 */
#include <Atom/RHI/FrameGraph.h>
#include <AzCore/std/parallel/binary_semaphore.h>
#include <AzCore/std/parallel/thread.h>
#include <RHI/CommandQueueContext.h>
#include <RHI/Device.h>
#include <RHI/FrameGraphExecuteGroupPrimary.h>
#include <RHI/FrameGraphExecuteGroupPrimaryHandler.h>
#include <RHI/FrameGraphExecuteGroupSecondary.h>
#include <RHI/FrameGraphExecuteGroupSecondaryHandler.h>
#include <RHI/FrameGraphExecuter.h>
#include <RHI/Scope.h>
#include <RHI/SwapChain.h>

namespace AZ
{
    namespace Vulkan
    {

        RHI::Ptr<FrameGraphExecuter> FrameGraphExecuter::Create()
        {
            return aznew FrameGraphExecuter();
        }

        FrameGraphExecuter::FrameGraphExecuter()
        {
            RHI::JobPolicy graphJobPolicy = RHI::JobPolicy::Parallel;
#if defined(AZ_FORCE_CPU_GPU_INSYNC)
            graphJobPolicy = RHI::JobPolicy::Serial;
#endif
            SetJobPolicy(graphJobPolicy);
        }

        RHI::ResultCode FrameGraphExecuter::InitInternal(const RHI::FrameGraphExecuterDescriptor& descriptor)
        {
            for (auto& [deviceIndex, platformLimitsDescriptor] : descriptor.m_platformLimitsDescriptors)
            {
                const RHI::ConstPtr<RHI::PlatformLimitsDescriptor> rhiPlatformLimitsDescriptor = platformLimitsDescriptor;
                if (RHI::ConstPtr<PlatformLimitsDescriptor> vkPlatformLimitsDesc =
                        azrtti_cast<const PlatformLimitsDescriptor*>(rhiPlatformLimitsDescriptor))
                {
                    m_frameGraphExecuterData[deviceIndex] = vkPlatformLimitsDesc->m_frameGraphExecuterData;
                }
            }

            return RHI::ResultCode::Success;
        }

        void FrameGraphExecuter::ShutdownInternal()
        {
            // do nothing
        }

        void FrameGraphExecuter::BeginInternal(const RHI::FrameGraph& frameGraph)
        {
            AZStd::vector<const Scope*> mergedScopes;
            const Scope* scopePrev = nullptr;
            const Scope* scopeNext = nullptr;
            const AZStd::vector<RHI::Scope*>& scopes = frameGraph.GetScopes();

            // The following semaphore trackers are there to count how many semaphores are present before each swapchain
            // Swapchains need to use a binary semaphore, which needs all dependent semaphores to be signalled before submitted to the queue
            // We do this by counting the number of semaphores that the scopes are waiting for
            // Not needed when AZ_FORCE_CPU_GPU_INSYNC because of synchronization after every scope
            AZStd::intrusive_ptr<SemaphoreTrackerCollection> semaphoreTrackers;
            // Tracker that will be used for the next swapchain in the framegraph
            AZStd::shared_ptr<SemaphoreTrackerHandle> currentSemaphoreHandle;
            // Some semaphores (Fences) might be waited-for by the use in a scope
            // We remember which semaphores are signalled and assume that the ones that are never waited-for are waited-for by the user
            AZStd::unordered_map<Fence*, bool> userFencesSignalledMap;
            [[maybe_unused]] int numUnwaitedFences = 0;
            bool useSemaphoreTrackers = frameGraph.GetScopes().front()->GetDevice().GetFeatures().m_signalFenceFromCPU;
            if (useSemaphoreTrackers)
            {
                semaphoreTrackers = new SemaphoreTrackerCollection;
                currentSemaphoreHandle = semaphoreTrackers->CreateHandle();
            }

#if defined(AZ_FORCE_CPU_GPU_INSYNC)
            // Forces all scopes to issue a dedicated merged scope group with one command list.
            // This will ensure that the Execute is done on only one scope and if an error happens
            // we can be sure about the work gpu was working on before the crash.
            for (auto it = scopes.begin(); it != scopes.end(); ++it)
            {
                const Scope& scope = *static_cast<const Scope*>(*it);
                auto nextIter = it + 1;
                scopeNext = nextIter != scopes.end() ? static_cast<const Scope*>(*nextIter) : nullptr;
                const bool subpassGroup = (scopeNext && scopeNext->GetFrameGraphGroupId() == scope.GetFrameGraphGroupId()) ||
                    (scopePrev && scopePrev->GetFrameGraphGroupId() == scope.GetFrameGraphGroupId());

                if (subpassGroup)
                {
                    FrameGraphExecuteGroupSecondary* scopeContextGroup = AddGroup<FrameGraphExecuteGroupSecondary>();
                    scopeContextGroup->Init(static_cast<Device&>(scope.GetDevice()), scope, 1, GetJobPolicy(), nullptr);
                }
                else
                {
                    mergedScopes.push_back(&scope);
                    FrameGraphExecuteGroupPrimary* multiScopeContextGroup = AddGroup<FrameGraphExecuteGroupPrimary>();
                    multiScopeContextGroup->SetName(scope.GetName());
                    multiScopeContextGroup->Init(static_cast<Device&>(scope.GetDevice()), AZStd::move(mergedScopes), nullptr);
                }
                scopePrev = &scope;
            }
#else

            RHI::HardwareQueueClass mergedHardwareQueueClass = RHI::HardwareQueueClass::Graphics;
            uint32_t mergedGroupCost = 0;
            uint32_t mergedSwapchainCount = 0;
            int mergedDeviceIndex = RHI::MultiDevice::InvalidDeviceIndex;

            for (auto it = scopes.begin(); it != scopes.end(); ++it)
            {
                const Scope& scope = *static_cast<const Scope*>(*it);
                auto nextIter = it + 1;
                scopeNext = nextIter != scopes.end() ? static_cast<const Scope*>(*nextIter) : nullptr;

                // Reset merged hardware queue class to match current scope if empty.
                if (mergedGroupCost == 0)
                {
                    mergedHardwareQueueClass = scope.GetHardwareQueueClass();
                }

                const uint32_t estimatedItemCount = scope.GetEstimatedItemCount();

                const uint32_t CommandListCostThreshold = AZStd::max(
                    m_frameGraphExecuterData[scope.GetDeviceIndex()].m_commandListCostThresholdMin,
                    AZ::DivideAndRoundUp(estimatedItemCount, m_frameGraphExecuterData[scope.GetDeviceIndex()].m_commandListsPerScopeMax));

                /**
                 * Computes a cost heuristic based on the number of items and number of attachments in
                 * the scope. This cost is used to partition command list generation.
                 */
                const uint32_t totalScopeCost = estimatedItemCount * m_frameGraphExecuterData[scope.GetDeviceIndex()].m_itemCost +
                    static_cast<uint32_t>(scope.GetAttachments().size()) *
                        m_frameGraphExecuterData[scope.GetDeviceIndex()].m_attachmentCost;

                // Check if we are in a middle of a framegraph group.
                const bool subpassGroup = (scopeNext && scopeNext->GetFrameGraphGroupId() == scope.GetFrameGraphGroupId()) ||
                    (scopePrev && scopePrev->GetFrameGraphGroupId() == scope.GetFrameGraphGroupId());

                const uint32_t swapchainCount = static_cast<uint32_t>(scope.GetSwapChainsToPresent().size());

                // Detect if we are able to continue merging.
                {
                    // Check if the group fits into the current running merge queue. If not, we have to flush the queue.
                    const bool exceededCommandCost = (mergedGroupCost + totalScopeCost) > CommandListCostThreshold;

                    // Check if the swap chains fit into this group.
                    const bool exceededSwapChainLimit = (mergedSwapchainCount + swapchainCount) >
                        m_frameGraphExecuterData[scope.GetDeviceIndex()].m_swapChainsPerCommandList;

                    // Check if the hardware queue classes match.
                    const bool hardwareQueueMismatch = scope.GetHardwareQueueClass() != mergedHardwareQueueClass;

                    // Check if we are straddling the boundary of a fence/semaphore.
                    const bool onSyncBoundaries = !scope.GetWaitSemaphores().empty() || !scope.GetWaitFences().empty() ||
                        (scopePrev && (!scopePrev->GetSignalSemaphores().empty() || !scopePrev->GetSignalFences().empty()));

                    // Check if the devices match.
                    const bool deviceMismatch = mergedDeviceIndex != scope.GetDeviceIndex();

                    // If we exceeded limits, then flush the group.
                    const bool flushMergedScopes = exceededCommandCost || exceededSwapChainLimit || hardwareQueueMismatch ||
                        onSyncBoundaries || deviceMismatch || subpassGroup;

                    if (flushMergedScopes && mergedScopes.size())
                    {
                        // All merged scopes use a single primary command list
                        mergedGroupCost = 0;
                        mergedSwapchainCount = 0;
                        mergedHardwareQueueClass = scope.GetHardwareQueueClass();
                        mergedDeviceIndex = scope.GetDeviceIndex();
                        FrameGraphExecuteGroupPrimary* multiScopeContextGroup = AddGroup<FrameGraphExecuteGroupPrimary>();
                        multiScopeContextGroup->Init(
                            static_cast<Device&>(scopePrev->GetDevice()), AZStd::move(mergedScopes), currentSemaphoreHandle);
                    }
                }

                if (useSemaphoreTrackers)
                {
                    if (scopePrev && !scopePrev->GetSwapChainsToPresent().empty())
                    {
                        currentSemaphoreHandle = semaphoreTrackers->CreateHandle();
                    }
                    for (auto& fence : scope.GetSignalFences())
                    {
                        auto it = userFencesSignalledMap.find(fence.get());
                        if (it == userFencesSignalledMap.end())
                        {
                            userFencesSignalledMap[fence.get()] = false;
                            numUnwaitedFences++;
                        }
                    }
                    for (auto& fence : scope.GetWaitFences())
                    {
                        auto it = userFencesSignalledMap.find(fence.get());
                        if (it != userFencesSignalledMap.end())
                        {
                            if (!it->second)
                            {
                                numUnwaitedFences--;
                            }
                        }
                        userFencesSignalledMap[fence.get()] = true;
                    }
                    semaphoreTrackers->AddSemaphores(scope.GetWaitSemaphores().size() + scope.GetWaitFences().size());

                    for (auto& swapchain : scope.GetSwapChainsToPresent())
                    {
                        semaphoreTrackers->AddSemaphores(numUnwaitedFences);
                        numUnwaitedFences = 0;
                        userFencesSignalledMap.clear();
                        // TODO no need to create a separate tracker for multiple swap chains in the same group
                        auto vulkanSwapChain = static_cast<SwapChain*>(swapchain);
                        vulkanSwapChain->SetSemaphoreTracker(semaphoreTrackers->GetCurrentTracker());
                    }
                }

                // Attempt to merge the current scope.
                if (!subpassGroup && totalScopeCost < CommandListCostThreshold)
                {
                    mergedScopes.push_back(&scope);
                    mergedGroupCost += totalScopeCost;
                    mergedSwapchainCount += swapchainCount;
                }
                // Not mergeable, create a dedicated context group for it.
                else
                {
                    // And then create a new group for the current scope with dedicated [1, N] secondary command lists
                    const uint32_t commandListCount = AZStd::max(AZ::DivideAndRoundUp(totalScopeCost, CommandListCostThreshold), 1u);
                    FrameGraphExecuteGroupSecondary* scopeContextGroup = AddGroup<FrameGraphExecuteGroupSecondary>();
                    scopeContextGroup->Init(
                        static_cast<Device&>(scope.GetDevice()), scope, commandListCount, GetJobPolicy(), currentSemaphoreHandle);
                }
                scopePrev = &scope;
            }

            // Merge all pending scopes
            if (mergedScopes.size())
            {
                mergedGroupCost = 0;
                mergedSwapchainCount = 0;
                FrameGraphExecuteGroupPrimary* multiScopeContextGroup = AddGroup<FrameGraphExecuteGroupPrimary>();
                multiScopeContextGroup->Init(
                    static_cast<Device&>(mergedScopes.front()->GetDevice()), AZStd::move(mergedScopes), currentSemaphoreHandle);
            }
#endif
            // Create the handlers to manage the execute groups.
            // Handlers manage one or multiple execute groups by creating a shared renderpass/framebuffer
            // or advancing the subpass if needed.
            auto groups = GetGroups();
            AZStd::vector<RHI::FrameGraphExecuteGroup*> groupRefs;
            groupRefs.reserve(groups.size());
            RHI::GraphGroupId groupId;
            uint32_t initGroupIndex = 0;
            for (uint32_t i = 0; i < groups.size(); ++i)
            {
                const FrameGraphExecuteGroup* group = static_cast<const FrameGraphExecuteGroup*>(groups[i].get());
                if (groupId != group->GetGroupId())
                {
                    groupRefs.clear();
                    for (size_t groupRefIndex = initGroupIndex; groupRefIndex < i; ++groupRefIndex)
                    {
                        groupRefs.push_back(groups[groupRefIndex].get());
                    }
                    AddExecuteGroupHandler(groupId, groupRefs);
                    groupId = group->GetGroupId();
                    initGroupIndex = i;
                }
            }

            // Add the final handler for the remaining groups.
            groupRefs.clear();
            for (size_t groupRefIndex = initGroupIndex; groupRefIndex < groups.size(); ++groupRefIndex)
            {
                groupRefs.push_back(groups[groupRefIndex].get());
            }
            AddExecuteGroupHandler(groupId, groupRefs);
        }

        void FrameGraphExecuter::ExecuteGroupInternal(RHI::FrameGraphExecuteGroup& groupBase)
        {
            FrameGraphExecuteGroup& group = static_cast<FrameGraphExecuteGroup&>(groupBase);
            auto findIter = m_groupHandlers.find(group.GetGroupId());
            AZ_Assert(findIter != m_groupHandlers.end(), "Could not find group handler for groupId %d", group.GetGroupId().GetIndex());
            FrameGraphExecuteGroupHandler* handler = findIter->second.get();
            // Wait until all execute groups of the handler has finished and also make sure that the handler itself hasn't executed already
            // (which is possible for parallel encoding).
            if (!handler->IsExecuted() && handler->IsComplete())
            {
                // This will execute the recorded work into the queue.
                handler->End();
            }
        }

        void FrameGraphExecuter::EndInternal()
        {
            m_groupHandlers.clear();
        }

        void FrameGraphExecuter::AddExecuteGroupHandler(
            const RHI::GraphGroupId& groupId, const AZStd::vector<RHI::FrameGraphExecuteGroup*>& groups)
        {
            if (groups.empty())
            {
                return;
            }

            // Add the handler depending on the number of execute groups.
            AZStd::unique_ptr<FrameGraphExecuteGroupHandler> handler(
                groups.size() == 1 && azrtti_cast<FrameGraphExecuteGroupPrimary*>(groups.front())
                    ? static_cast<FrameGraphExecuteGroupHandler*>(aznew FrameGraphExecuteGroupPrimaryHandler)
                    : static_cast<FrameGraphExecuteGroupHandler*>(aznew FrameGraphExecuteGroupSecondaryHandler));

            auto firstGroup = static_cast<FrameGraphExecuteGroup*>(groups.front());
            handler->Init(static_cast<FrameGraphExecuteGroup*>(groups.front())->GetDevice(), groups, firstGroup->GetFenceTracker());
            m_groupHandlers.insert({ groupId, AZStd::move(handler) });
        }
    } // namespace Vulkan
} // namespace AZ
