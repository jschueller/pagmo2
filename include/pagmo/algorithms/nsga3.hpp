/* Copyright 2017-2021 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_ALGORITHMS_NSGA3_HPP
#define PAGMO_ALGORITHMS_NSGA3_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/rng.hpp>                   // random_device, random_engine_type
#include <pagmo/detail/visibility.hpp>     // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>            // population
#include <pagmo/utils/reference_point.hpp> // ReferencePoint


namespace pagmo{

/// Nondominated Sorting Genetic Algorithm III (NSGA-III)
/**
 * \image html nsga3.jpg "The NSGA-III flowchart" width=3cm
 *
 * NSGA-III is a reference-point-based multi-objective evolutionary algorithm designed
 * to handle many-objective optimization problems (problems with 3 or more objectives).
 * It extends the NSGA-II framework by replacing the crowding distance operator with
 * a niche-preservation operator based on uniformly distributed reference points on a
 * normalized hyperplane.
 *
 * The algorithm:
 * - Generates a set of uniformly distributed reference points on a unit simplex via the
 *   Das and Dennis (1998) systematic approach.
 * - Creates offspring via simulated binary crossover (SBX) and polynomial mutation.
 * - Combines parent and offspring populations (size 2N).
 * - Performs non-dominated sorting on the combined population.
 * - Selects N individuals for the next generation using reference-point-based niching,
 *   which associates each individual with its nearest reference point and prefers
 *   reference points with fewer associated members.
 * - Includes optional adaptive normalization using extreme points and hyperplane
 *   intercepts (Das & Dennis, 1998).
 *
 * See: Deb, K. and Jain, H., "An Evolutionary Many-Objective Optimization Algorithm
 * Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems
 * With Box Constraints," IEEE Transactions on Evolutionary Computation, vol. 18, no. 4,
 * pp. 577-601, Aug. 2014. DOI: 10.1109/TEVC.2013.2281535
 */
class PAGMO_DLL_PUBLIC nsga3{
    public:
        /// NSGA-III cross-generation memory for adaptive normalization
        /**
         * Stores the extreme points, ideal point, and nadir point from previous
         * generations to improve stability of the adaptive normalization (Section IV).
         */
        typedef struct{
            std::vector<std::vector<double>> v_extreme;
            std::vector<double> v_ideal;
            std::vector<double> v_nadir;
        } NSGA3Memory;
        /// Single entry of the log (gen, fevals, ideal_point)
        typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
        /// The log
        typedef std::vector<log_line_type> log_type;

        /// Constructor
        /**
         * Constructs the NSGA-III user-defined algorithm.
         *
         * Default parameter values are taken from Deb & Jain, 2014, Section V, Table I.
         *
         * @param[in] gen Number of generations to evolve.
         * @param[in] cr Crossover probability (SBX).
         * @param[in] eta_c Distribution index for SBX crossover.
         * @param[in] mut Mutation probability (polynomial).
         * @param[in] eta_mut Distribution index for polynomial mutation.
         * @param[in] divisions Number of divisions per objective for generating
         *                      reference points on the unit simplex.
         * @param[in] seed Seed used by the internal RNG (default is random).
         * @param[in] use_memory If true, preserves extreme, ideal, and nadir points
         *                       across generations for adaptive normalization stability.
         *
         * @throws std::invalid_argument if \p cr is not in [0,1],
         *         \p mut is not in [0,1], \p eta_c or \p eta_mut are not in [1,100],
         *         or \p divisions < 1.
         */
        nsga3(unsigned gen = 1u, double cr = 1.0, double eta_c = 30.0,
              double mut = 0.10, double eta_mut = 20.0, size_t divisions = 12u,
              unsigned seed = pagmo::random_device::next(), bool use_memory = false);

        /// Algorithm name
        /**
         * @return a string containing "NSGA-III:"
         */
        std::string get_name() const{ return "NSGA-III:"; }

        /// Extra info string
        /**
         * @return a string containing parameters: generations, crossover probability,
         *         crossover distribution index, mutation probability, mutation
         *         distribution index, reference point divisions, seed, verbosity.
         */
        std::string get_extra_info() const;

        // Algorithm evolve method
        population evolve(population) const;

        /// Niche-preserving selection operator
        /**
         * Implements the reference-point-based niching selection (Algorithm 1, Step 10-11).
         * Accepts all members of the first l-1 fronts, then uses niche counting and
         * the niching operator on the last front to select the remaining individuals.
         *
         * @param R Combined parent-offspring population (size 2*N_pop).
         * @param N_pop Target population size.
         * @return indices of selected individuals.
         */
        std::vector<size_t> selection(population &, size_t) const;

        /// Generate uniformly distributed reference points
        /**
         * Generates reference points on the unit simplex using the Das & Dennis (1998)
         * systematic approach, as described in Section V-A and Algorithm 1.
         *
         * The number of reference points generated is C(nobjs + divisions - 1, divisions).
         *
         * @param nobjs Number of objectives.
         * @param divisions Number of divisions per objective.
         * @return vector of ReferencePoint objects whose coefficients sum to 1.0.
         */
        std::vector<ReferencePoint> generate_uniform_reference_points(size_t nobjs, size_t divisions) const;

        /// Translate objectives by the ideal point (Algorithm 1, Step 4)
        /**
         * Subtracts the ideal point from each objective vector.
         * Optionally incorporates the ideal point from prior generations when
         * cross-generation memory is enabled.
         *
         * @param pop Population whose objectives are to be translated.
         * @return Translated objective vectors.
         */
        std::vector<std::vector<double>> translate_objectives(population) const;

        /// Compute extreme points for adaptive normalization (Algorithm 1, Step 5)
        /**
         * Determines the extreme point for each objective axis by minimizing
         * the Achievement Scalarization Function (ASF) over the first non-dominated
         * front and optionally prior extreme points from memory.
         *
         * @param pop The population.
         * @param fronts Non-dominated fronts from fast_non_dominated_sorting.
         * @param translated_objs Objective vectors translated by the ideal point.
         * @return A vector of n_obj extreme points (one per objective).
         */
        std::vector<std::vector<double>> find_extreme_points(population, std::vector<std::vector<pop_size_t>> &, std::vector<std::vector<double>> &) const;

        /// Compute hyperplane intercepts (Algorithm 1, Step 6)
        /**
         * Constructs a hyperplane from the extreme points by solving Ax = b
         * via Gaussian elimination. The reciprocals of the solution give the
         * intercepts. Falls back to the nadir point if the extreme points are
         * degenerate (duplicate points leading to a singular matrix) or if
         * any intercept is negative.
         *
         * @param pop The population.
         * @param ext_points Extreme points (one per objective).
         * @return Intercept values for each objective dimension.
         */
        std::vector<double> find_intercepts(population, std::vector<std::vector<double>> &) const;

        /// Normalize objectives using intercepts (Algorithm 1, Step 7, Eq. 4)
        /**
         * Divides each translated objective value by the corresponding intercept,
         * clamping the intercept to a minimum of \c std::numeric_limits<double>::epsilon()
         * to avoid division by zero.
         *
         * @param translated_objs Objective vectors previously translated by the ideal point.
         * @param intercepts Hyperplane intercepts for each objective.
         * @return Normalized objective vectors.
         */
        std::vector<std::vector<double>> normalize_objectives(std::vector<std::vector<double>> &, std::vector<double> &) const;

        /// Sets the algorithm verbosity
        /**
         * Sets the verbosity level of screen output and of the log returned by get_log().
         * \p level can be:
         * - 0: no verbosity
         * - >0: will print and log one line each \p level generations.
         *
         * Example (verbosity 1):
         * @code{.unparsed}
         * Gen:        Fevals:        ideal1:        ideal2:        ideal3:
         *   1              0      0.203721      0.0584764        4.09321
         * @endcode
         *
         * @param level Verbosity level.
         */
        void set_verbosity(unsigned level) { m_verbosity = level; }

        /// Gets the verbosity level
        /**
         * @return the verbosity level.
         */
        unsigned get_verbosity() const { return m_verbosity; }

        /// Gets the seed
        /**
         * @return the seed controlling the algorithm's stochastic behaviour.
         */
        unsigned get_seed() const { return m_seed; }

        /// Sets the seed
        /**
         * @param seed New seed for the internal RNG.
         */
        void set_seed(unsigned seed) { m_reng.seed(seed); m_seed = seed; }

        /// Get log
        /**
         * A log containing relevant quantities monitoring the last call to evolve.
         * Each element of the returned vector contains: Gen, Fevals, ideal_point
         * as described in set_verbosity.
         *
         * @return an std::vector of log_line_type containing the logged values.
         */
        const log_type &get_log() const { return m_log; }

        /// Check whether cross-generation memory is enabled
        /**
         * @return true if extreme/ideal/nadir points are preserved across generations.
         */
        bool has_memory() const {return m_use_memory; }

    private:
        unsigned m_gen;
        double m_cr;
        double m_eta_c;
        double m_mut;
        double m_eta_mut;
        size_t m_divisions;
        unsigned m_seed;
        bool m_use_memory;
        mutable NSGA3Memory m_memory{};
        mutable detail::random_engine_type m_reng;
        mutable log_type m_log;
        unsigned m_verbosity {0};
        mutable std::vector<ReferencePoint> m_refpoints;
        // Serialisation support
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &, unsigned int);
};

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga3)
#endif
