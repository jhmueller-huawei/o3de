/*
 * Copyright (c) Contributors to the Open 3D Engine Project.
 * For complete copyright and license terms please see the LICENSE at the root of this distribution.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 */
#pragma once

#include <Atom/RHI/SingleDeviceImageView.h>
#include <AzCore/Memory/PoolAllocator.h>

namespace AZ
{
    namespace Null
    {
        class Image;

        class ImageView final
            : public RHI::SingleDeviceImageView
        {
            using Base = RHI::SingleDeviceImageView;
        public:
            AZ_CLASS_ALLOCATOR(ImageView, AZ::ThreadPoolAllocator);
            AZ_RTTI(ImageView, "{4960F7E4-1B2D-4F35-97B3-7B21A5E1C516}", Base);

            static RHI::Ptr<ImageView> Create();

        private:
            ImageView() = default;

            //////////////////////////////////////////////////////////////////////////
            // RHI::SingleDeviceImageView
            RHI::ResultCode InitInternal([[maybe_unused]] RHI::Device& device, [[maybe_unused]] const RHI::SingleDeviceResource& resourceBase) override { return RHI::ResultCode::Success;}
            RHI::ResultCode InvalidateInternal() override { return RHI::ResultCode::Success;}
            void ShutdownInternal() override {}
            //////////////////////////////////////////////////////////////////////////
        };
    }
}
