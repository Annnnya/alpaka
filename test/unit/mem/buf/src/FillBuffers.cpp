/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

TEMPLATE_LIST_TEST_CASE("memBufFillTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test fill function");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    constexpr Elem fillVal = 42;
    alpaka::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = alpaka::allocBuf<Elem, Idx>(devHost, extent);
    alpaka::memcpy(queue, bufHost, buf);
    alpaka::wait(queue);

    Elem const* ptr = std::data(bufHost);
    Idx const size = alpaka::getExtentProduct(bufHost);
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(ptr[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}
