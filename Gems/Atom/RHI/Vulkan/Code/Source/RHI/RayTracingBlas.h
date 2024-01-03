/*
 * Copyright (c) Contributors to the Open 3D Engine Project.
 * For complete copyright and license terms please see the LICENSE at the root of this distribution.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 */
#pragma once

#include <Atom_RHI_Vulkan_Platform.h>
#include <Atom/RHI/SingleDeviceRayTracingAccelerationStructure.h>
#include <AzCore/Memory/SystemAllocator.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>

namespace AZ
{
    namespace Vulkan
    {
        class Buffer;

        //! This class builds and contains the Vulkan RayTracing BLAS buffers.
        class RayTracingBlas final
            : public RHI::SingleDeviceRayTracingBlas
        {
        public:
            AZ_CLASS_ALLOCATOR(RayTracingBlas, AZ::SystemAllocator);

            static RHI::Ptr<RayTracingBlas> Create();

            struct BlasBuffers
            {
                RHI::Ptr<RHI::SingleDeviceBuffer> m_blasBuffer;
                RHI::Ptr<RHI::SingleDeviceBuffer> m_scratchBuffer;
                VkAccelerationStructureKHR m_accelerationStructure = VK_NULL_HANDLE;

                AZStd::vector<VkAccelerationStructureGeometryKHR> m_geometryDescs;
                AZStd::vector<VkAccelerationStructureBuildRangeInfoKHR> m_rangeInfos;
                VkAccelerationStructureBuildGeometryInfoKHR m_buildInfo = {};
            };

            const BlasBuffers& GetBuffers() const { return m_buffers[m_currentBufferIndex]; }

            // RHI::SingleDeviceRayTracingBlas overrides...
            virtual bool IsValid() const override { return m_buffers[m_currentBufferIndex].m_accelerationStructure != VK_NULL_HANDLE; }

        private:
            RayTracingBlas() = default;

            // RHI::SingleDeviceRayTracingBlas overrides...
            RHI::ResultCode CreateBuffersInternal(RHI::Device& deviceBase, const RHI::SingleDeviceRayTracingBlasDescriptor* descriptor, const RHI::SingleDeviceRayTracingBufferPools& rayTracingBufferPools) override;

            static VkBuildAccelerationStructureFlagsKHR GetAccelerationStructureBuildFlags(const RHI::RayTracingAccelerationStructureBuildFlags &buildFlags);

            // buffer list to keep buffers alive for several frames
            static const uint32_t BufferCount = 3;
            BlasBuffers m_buffers[BufferCount];
            uint32_t m_currentBufferIndex = 0;
        };
    }
}
