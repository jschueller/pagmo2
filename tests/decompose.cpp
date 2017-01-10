#define BOOST_TEST_MODULE decompose_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <limits>
#include <stdexcept>
#include <string>

#include "../include/io.hpp"
#include "../include/problems/decompose.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/types.hpp"

using namespace pagmo;

struct mc_01 {
    vector_double fitness(const vector_double &) const
    {
        return {1., 1.};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

BOOST_AUTO_TEST_CASE(decompose_construction_test)
{
    // First we check directly the two constructors
    problem p0{decompose{}};
    problem p1{decompose{zdt{1u, 2u}, {0.5, 0.5}, {0., 0.}, "weighted", false}};

    auto p0_string = boost::lexical_cast<std::string>(p0);
    auto p1_string = boost::lexical_cast<std::string>(p1);

    // We check that the default constructor constructs a problem
    // which has an identical representation to the problem
    // built by the explicit constructor.
    BOOST_CHECK(p0_string == p1_string);

    // We check the throws
    auto inf = std::numeric_limits<double>::infinity();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    // single objective problem
    BOOST_CHECK_THROW(decompose(rosenbrock{}, {0.5, 0.5}, {0., 0.}), std::invalid_argument);
    // constrained problem
    BOOST_CHECK_THROW(decompose(mc_01{}, {0.5, 0.5}, {0., 0.}, "weighted", false), std::invalid_argument);
    // random decomposition method
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.5, 0.5}, {0., 0.}, "my_method", false), std::invalid_argument);
    // wrong length for the weights
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.5, 0.2, 0.3}, {0., 0.}, "weighted", false), std::invalid_argument);
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.5, inf}, {0., 0.}, "weighted", false), std::invalid_argument);
    // wrong length for the reference point
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.5, 0.5}, {1.}, "weighted", false), std::invalid_argument);
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.5, 0.5}, {0., nan}, "weighted", false), std::invalid_argument);
    // weight sum != 1
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {0.9, 0.5}, {0., 0.}, "weighted", false), std::invalid_argument);
    // weight contains negative component
    BOOST_CHECK_THROW(decompose(zdt{1u, 2u}, {1.5, -0.5}, {0., 0.}, "weighted", false), std::invalid_argument);

    print(p1);
}

BOOST_AUTO_TEST_CASE(decompose_integration_into_problem_test)
{
    problem p{decompose{zdt{1u, 2u}, {0.5, 0.5}, {0., 0.}, "weighted", false}};
    BOOST_CHECK(p.has_gradient() == false);
    BOOST_CHECK(p.has_hessians() == false);
    BOOST_CHECK(p.get_nobj() == 1u);
    BOOST_CHECK_THROW(p.gradient({1, 2}), std::logic_error);
    BOOST_CHECK_THROW(p.hessians({1, 2}), std::logic_error);
}

BOOST_AUTO_TEST_CASE(decompose_fitness_test)
{
    problem p{zdt{1u, 2u}};
    vector_double lambda{0.5, 0.5};
    vector_double z{0., 0.};
    problem pdw{decompose{zdt{1u, 2u}, lambda, z, "weighted", false}};
    problem pdtch{decompose{zdt{1u, 2u}, lambda, z, "tchebycheff", false}};
    problem pdbi{decompose{zdt{1u, 2u}, lambda, z, "bi", false}};

    vector_double point{1., 1.};
    auto f = p.fitness(point);
    auto fdw = pdw.fitness(point);
    auto fdtch = pdtch.fitness(point);
    auto fdbi = pdbi.fitness(point);

    BOOST_CHECK_CLOSE(fdw[0], f[0] * lambda[0] + f[1] * lambda[1], 1e-8);
    BOOST_CHECK_CLOSE(fdtch[0], std::max(lambda[0] * std::abs(f[0] - z[0]), lambda[1] * std::abs(f[1] - z[1])), 1e-8);
    double lnorm = std::sqrt(lambda[0] * lambda[0] + lambda[1] * lambda[1]);
    vector_double ilambda{lambda[0] / lnorm, lambda[1] / lnorm};
    double d1 = (f[0] - z[0]) * ilambda[0] + (f[1] - z[1]) * ilambda[1];
    double d20 = f[0] - (z[0] + d1 * ilambda[0]);
    double d21 = f[1] - (z[1] + d1 * ilambda[1]);
    d20 *= d20;
    d21 *= d21;
    double d2 = std::sqrt(d20 + d21);
    BOOST_CHECK_CLOSE(fdbi[0], d1 + 5.0 * d2, 1e-8);
}

BOOST_AUTO_TEST_CASE(original_and_decomposed_fitness_test)
{
    zdt p{zdt{1u, 2u}};
    vector_double lambda{0.5, 0.5};
    vector_double z{0., 0.};
    decompose pdw{zdt{1u, 2u}, lambda, z, "weighted", false};
    decompose pdtch{zdt{1u, 2u}, lambda, z, "tchebycheff", false};
    decompose pdbi{zdt{1u, 2u}, lambda, z, "bi", false};

    vector_double dv{1., 1.};
    auto f = p.fitness(dv);
    auto fdw = pdw.original_fitness(dv);
    auto fdtch = pdtch.original_fitness(dv);
    auto fdbi = pdbi.original_fitness(dv);

    // We check that the original fitness is always the same
    BOOST_CHECK(f == fdw);
    BOOST_CHECK(f == fdtch);
    BOOST_CHECK(f == fdbi);

    // We check that the decomposed fitness is correct (by a direct call, not via fitness)
    lambda = {0.2, 0.8};
    z = {0.1, 0.1};
    fdw = pdw.decompose_fitness(f, lambda, z);
    fdtch = pdtch.decompose_fitness(f, lambda, z);
    fdbi = pdbi.decompose_fitness(f, lambda, z);
    BOOST_CHECK_CLOSE(fdw[0], f[0] * lambda[0] + f[1] * lambda[1], 1e-8);
    BOOST_CHECK_CLOSE(fdtch[0], std::max(lambda[0] * std::abs(f[0] - z[0]), lambda[1] * std::abs(f[1] - z[1])), 1e-8);
    double lnorm = std::sqrt(lambda[0] * lambda[0] + lambda[1] * lambda[1]);
    vector_double ilambda{lambda[0] / lnorm, lambda[1] / lnorm};
    double d1 = (f[0] - z[0]) * ilambda[0] + (f[1] - z[1]) * ilambda[1];
    double d20 = f[0] - (z[0] + d1 * ilambda[0]);
    double d21 = f[1] - (z[1] + d1 * ilambda[1]);
    d20 *= d20;
    d21 *= d21;
    double d2 = std::sqrt(d20 + d21);
    BOOST_CHECK_CLOSE(fdbi[0], d1 + 5.0 * d2, 1e-8);

    // We check the throws
    BOOST_CHECK_THROW(pdw.decompose_fitness(f, {1., 2., 3}, z), std::invalid_argument);
    BOOST_CHECK_THROW(pdw.decompose_fitness(f, lambda, {1.}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(decompose_ideal_point_adaptation_test)
{
    // no adaptation
    {
        problem p{decompose{zdt{1u, 2u}, {0.5, 0.5}, {2., 2.}, "weighted", false}};
        BOOST_CHECK(p.extract<decompose>()->get_z() == vector_double({2., 2.}));
        p.fitness({1., 1.});
        BOOST_CHECK(p.extract<decompose>()->get_z() == vector_double({2., 2.}));
    }

    // adaptation at work
    {
        problem p{decompose{zdt{1u, 2u}, {0.5, 0.5}, {2., 2.}, "weighted", true}};
        BOOST_CHECK(p.extract<decompose>()->get_z() == vector_double({2., 2.}));
        p.fitness({1., 1.});
        BOOST_CHECK(p.extract<decompose>()->get_z() == vector_double({1., 2.}));
        p.fitness({0., 0.});
        BOOST_CHECK(p.extract<decompose>()->get_z() == vector_double({0., 1.}));
    }
}

BOOST_AUTO_TEST_CASE(decompose_has_dense_sparsities_test)
{
    problem p{decompose{zdt{1u, 2u}, {0.5, 0.5}, {2., 2.}, "weighted", false}};
    BOOST_CHECK(p.gradient_sparsity() == detail::dense_gradient(1u, 2u));
    BOOST_CHECK(p.hessians_sparsity() == detail::dense_hessians(1u, 2u));
}

BOOST_AUTO_TEST_CASE(decompose_name_and_extra_info_test)
{
    decompose p{zdt{1u, 2u}, {0.5, 0.5}, {2., 2.}, "weighted", false};
    BOOST_CHECK(p.get_name().find("[decomposed]") != std::string::npos);
    BOOST_CHECK(p.get_extra_info().find("Ideal point adaptation") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(decompose_serialization_test)
{
    problem p{decompose{zdt{1u, 2u}, {0.5, 0.5}, {2., 2.}, "weighted", false}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1.});
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{null_problem{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
}