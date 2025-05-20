#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/NdLoop.hpp"

namespace alpaka
{
    class DevCpu;

    namespace detail
    {
        // Base for Fill task
        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillCpuBase
        {
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            static_assert(std::is_same_v<Elem, TValue>, "Fill value type must match view element type");
            static_assert(std::is_trivially_copyable_v<Elem>, "Only trivially copyable types supported for fill");

            template<typename TViewFwd>
            TaskFillCpuBase(TViewFwd&& view, TValue const& value, TExtent const& extent)
                : m_value(value)
                , m_extent(getExtents(extent))
                , m_extentWidth(getExtents(extent).back())
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtents(view))
#endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 1)
                    ALPAKA_ASSERT(
                        m_extentWidth * static_cast<ExtentSize>(sizeof(Elem)) <= m_dstPitchBytes[TDim::value - 2]);

                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            TValue const m_value;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidth;
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            std::uint8_t* const m_dstMemNative;
        };

        // Generic ND version
        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillCpu : public TaskFillCpuBase<TDim, TView, TExtent, TValue>
        {
            using Base = TaskFillCpuBase<TDim, TView, TExtent, TValue>;
            using Base::Base;
            using Elem = typename Base::Elem;
            using ExtentSize = typename Base::ExtentSize;

            // using DstSize = typename Base::DstSize;
            // using DimMin1 = DimInt<TDim::value - 1u>;

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                // Vec<DimMin1, ExtentSize> const extentWithoutInnermost = subVecBegin<DimMin1>(this->m_extent);
                // Vec<DimMin1, DstSize> const pitchBytesWithoutOutmost = subVecBegin<DimMin1>(this->m_dstPitchBytes);

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        this->m_extent,
                        [&](Vec<TDim, ExtentSize> const& idx)
                        {
                            std::uintptr_t offsetBytes
                                = static_cast<std::uintptr_t>((idx * this->m_dstPitchBytes).sum());
                            assert(offsetBytes % alignof(Elem) == 0);
                            Elem* elem = reinterpret_cast<Elem*>(
                                __builtin_assume_aligned(this->m_dstMemNative + offsetBytes, alignof(Elem)));

                            *elem = this->m_value;
                        });
                }
            }
        };

        // 0D version (scalar fill)
        template<typename TView, typename TExtent, typename TValue>
        struct TaskFillCpu<DimInt<0u>, TView, TExtent, TValue>
        {
            using Elem = alpaka::Elem<TView>;

            static_assert(std::is_same_v<Elem, TValue>, "Fill value must match view element type");

            template<typename TViewFwd>
            TaskFillCpu(TViewFwd&& view, TValue const& value, [[maybe_unused]] TExtent const& extent)
                : m_value(value)
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            ALPAKA_FN_HOST auto operator()() const noexcept -> void
            {
                *reinterpret_cast<Elem*>(m_dstMemNative) = m_value;
            }

            TValue const m_value;
            std::uint8_t* const m_dstMemNative;
        };
    } // namespace detail

    namespace trait
    {
        //! The memory fill task trait specialization for CPU devices.
        template<typename TDim>
        struct CreateTaskFill<TDim, DevCpu>
        {
            template<typename TExtent, typename TViewFwd, typename TValue>
            ALPAKA_FN_HOST static auto createTaskFill(TViewFwd&& view, TValue const& value, TExtent const& extent)
            {
                using TView = std::remove_reference_t<TViewFwd>;
                static_assert(
                    std::is_same_v<TValue, alpaka::Elem<TView>>,
                    "Fill value type must match view element type");
                static_assert(
                    std::is_trivially_copyable_v<TValue>,
                    "Only trivially copyable types are supported for fill");

                return alpaka::detail::TaskFillCpu<TDim, TView, TExtent, TValue>{
                    std::forward<TViewFwd>(view),
                    value,
                    extent};
            }
        };
    } // namespace trait

} // namespace alpaka
