/*
 *  Defines the ReferencePoint type and related utilities used by the
 *  NSGA-III algorithm (Deb & Jain, 2014).
 *
 */

#ifndef PAGMO_UTILS_REFERENCE_POINT
#define PAGMO_UTILS_REFERENCE_POINT

#include <iostream>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>         // pop_size_t


namespace pagmo{

/// Reference point for the NSGA-III niching operator
/**
 * Represents a single reference point (direction vector) on the unit simplex
 * used by NSGA-III for niche-preserving selection (Deb & Jain, 2014, Section V-A).
 *
 * Each reference point:
 * - Has coefficients on the unit simplex (sum to 1.0).
 * - Maintains a count of associated members from the first l-1 non-dominated fronts.
 * - Maintains a list of candidate individuals from the last front, each with
 *   their perpendicular distance to the reference point.
 * - Supports the niching operator via select_member(), nearest_candidate(),
 *   and random_candidate() (Section IV-E).
 */
class PAGMO_DLL_PUBLIC ReferencePoint{
    public:
        /// Constructor
        /**
         * @param nobj Dimensionality of the reference point (number of objectives).
         */
        ReferencePoint(size_t nobj);

        /// Returns the dimensionality of the reference point
        size_t dim() const;

        /// Access coefficient at index \p idx
        double& operator[](size_t);

        /// Stream insertion operator
        friend PAGMO_DLL_PUBLIC std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp);

        /// Increment the member count (rho_j++)
        void increment_members(){ ++nmembers; }

        /// Decrement the member count
        void decrement_members(){ --nmembers; }

        /// Returns the number of associated members from the first l-1 fronts
        size_t member_count() const{ return nmembers; }

        /// Add a candidate from the last front
        /**
         * @param index Population index of the candidate.
         * @param distance Perpendicular distance to this reference point.
         */
        void add_candidate(size_t, double);

        /// Remove a candidate by population index
        void remove_candidate(size_t index);

        /// Returns the number of candidates from the last front
        size_t candidate_count() const{ return candidates.size(); }

        /// Returns the reference point coefficients
        std::vector<double> get_coeffs(){ return coeffs; }

        /// Returns the nearest candidate from the last front (minimum perpendicular distance)
        /**
         * Section IV-E: used when rho_j == 0 to select the closest candidate.
         */
        std::optional<size_t> nearest_candidate() const;

        /// Returns a random candidate from the last front
        /**
         * Section IV-E: used when rho_j >= 1 to randomly select a candidate.
         */
        std::optional<size_t> random_candidate() const;

        /// Niching operator
        /**
         * Selects one candidate from the last front according to the niche
         * preservation rule (Section IV-E): if no members (rho_j == 0),
         * return the nearest candidate; otherwise return a random candidate.
         *
         * @return Optional population index of the selected candidate,
         *         or std::nullopt if no candidates exist.
         */
        std::optional<size_t> select_member() const;
    protected:
        std::vector<double> coeffs{0};
        size_t nmembers{0};
        std::vector<std::pair<size_t, double>> candidates;
};

/// Generate reference points on the unit simplex recursively
/**
 * Generates reference points using the Das & Dennis (1998) systematic approach
 * (Deb & Jain, 2014, Algorithm 1, Step 3).
 *
 * @param rp Base reference point to populate recursively.
 * @param remain Remaining sum to allocate.
 * @param level Current objective index being populated.
 * @param total Total number of divisions (same as \p remain initially).
 * @return A vector of all reference points on the unit simplex.
 */
std::vector<ReferencePoint> generate_reference_point_level(
    ReferencePoint& rp,
    size_t remain,
    size_t level,
    size_t total
);

/// Associate individuals with reference points
/**
 * Associates each individual from the non-dominated fronts with its nearest
 * reference point (by perpendicular distance). For all fronts except the last,
 * the reference point's member count is incremented. For the last front,
 * individuals are stored as candidates with their distance (Deb & Jain, 2014,
 * Algorithm 1, Step 10).
 *
 * @param rps Reference points to associate with.
 * @param norm_objs Normalised objective vectors.
 * @param fronts Non-dominated fronts from fast non-dominated sorting.
 */
void associate_with_reference_points(
    std::vector<ReferencePoint> &,
    std::vector<std::vector<double>>,
    std::vector<std::vector<pop_size_t>>
);

/// Identify the niche with the fewest associated members
/**
 * Returns the index of the reference point with the minimum member count.
 * If multiple reference points share the minimum, one is chosen at random
 * (Deb & Jain, 2014, Algorithm 1, Step 11).
 *
 * @param rps Reference points to search.
 * @return Index of the selected niche reference point.
 */
size_t identify_niche_point(std::vector<ReferencePoint> &);

/// Binomial coefficient C(n, k)
/**
 * Computes the binomial coefficient \f$ \binom{n}{k} \f$ recursively.
 * This is used to compute the number of reference points:
 * \f$ C(M + p - 1, p) \f$ where M is the number of objectives and p is
 * the number of divisions (Das & Dennis, 1998).
 *
 * @param n Total number of elements.
 * @param k Number of elements to choose.
 * @return The binomial coefficient C(n, k).
 */
size_t n_choose_k(size_t, size_t);

}  // namespace pagmo

#endif
