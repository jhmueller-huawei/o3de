/*
 * Copyright (c) Contributors to the Open 3D Engine Project.
 * For complete copyright and license terms please see the LICENSE at the root of this distribution.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 */
#pragma once

#include "SemaphoreTracker.h"

namespace AZ
{
    namespace Vulkan
    {
        SemaphoreTracker::SemaphoreTracker(int initialNumberOfSemaphores)
            : m_semaphoreCounter(initialNumberOfSemaphores)
        {
        }
        void SemaphoreTracker::AddSemaphores(int numSemaphores)
        {
            m_semaphoreCounter += numSemaphores;
        }
        void SemaphoreTracker::SignalSemaphores(int numSemaphores)
        {
            AZStd::unique_lock<AZStd::mutex> lock(m_lock);
            m_signalledSemaphoreCounter += numSemaphores;
            if (m_signalledSemaphoreCounter == m_semaphoreCounter)
            {
                lock.unlock();
                m_waitCondition.notify_all();
            }
        }
        void SemaphoreTracker::WaitForSignalAllSemaphores()
        {
            AZStd::unique_lock<AZStd::mutex> lock(m_lock);
            m_waitCondition.wait(
                lock,
                [this]()
                {
                    return m_signalledSemaphoreCounter == m_semaphoreCounter;
                });
        }

        void SemaphoreTrackerCollection::AddSemaphores(int numSemaphores)
        {
            m_trackers.back()->AddSemaphores(numSemaphores);
            m_semaphoreCount += numSemaphores;
        }

        AZStd::shared_ptr<SemaphoreTrackerHandle> SemaphoreTrackerCollection::CreateHandle()
        {
            m_trackers.emplace_back(AZStd::make_unique<SemaphoreTracker>(m_semaphoreCount));
            auto result =
                AZStd::make_shared<SemaphoreTrackerHandle>(AZStd::intrusive_ptr<SemaphoreTrackerCollection>(this), m_trackers.size());
            return result;
        }

        const AZStd::shared_ptr<SemaphoreTracker>& SemaphoreTrackerCollection::GetCurrentTracker()
        {
            return m_trackers.back();
        };

        void SemaphoreTrackerCollection::SignalSemaphores(int countTrackers, int numSemphores)
        {
            for (int i = 0; i < countTrackers; i++)
            {
                m_trackers[i]->SignalSemaphores(numSemphores);
            }
        }

        SemaphoreTrackerHandle::SemaphoreTrackerHandle(AZStd::intrusive_ptr<SemaphoreTrackerCollection> tracker, int countTrackers)
            : m_countTrackers(countTrackers)
            , m_tracker(tracker)
        {
        }

        void SemaphoreTrackerHandle::SignalSemaphores(int numSemaphores)
        {
            m_tracker->SignalSemaphores(m_countTrackers, numSemaphores);
        }

    } // namespace Vulkan
} // namespace AZ
