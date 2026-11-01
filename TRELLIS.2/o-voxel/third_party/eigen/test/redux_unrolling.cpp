// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Reductions of a runtime size must not unroll, however large the unrolling budget grows. This limit reaches HugeCost,
// the cost of a runtime size, with or without vectorization. The default limit of 110 does so from 91 scalars per
// packet on (RVV at VLEN 1024 and LMUL 4), where the budget scales with the packet size.
#define EIGEN_UNROLLING_LIMIT 10000
#include "main.h"
#include "random_for_arithmetic.h"

template <typename Xpr>
typename Xpr::Scalar sum_by_coeff(const Xpr& x) {
  typename Xpr::Scalar sum(0);
  for (Index j = 0; j < x.cols(); ++j)
    for (Index i = 0; i < x.rows(); ++i) sum += x.coeff(i, j);
  return sum;
}

template <typename Xpr>
constexpr int sum_unrolling() {
  using Scalar = typename Xpr::Scalar;
  return internal::redux_traits<internal::scalar_sum_op<Scalar, Scalar>, internal::redux_evaluator<Xpr>>::Unrolling;
}

template <typename Panel>
constexpr int packetwise_sum_unrolling() {
  using Scalar = typename Panel::Scalar;
  return internal::packetwise_redux_traits<internal::scalar_sum_op<Scalar, Scalar>,
                                           internal::redux_evaluator<Panel>>::Unrolling;
}

template <typename Scalar>
void redux_runtime_size() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using ConstBlock = Block<const Mat>;
  using Panel = Block<const Mat, 4, Dynamic>;
  STATIC_CHECK(sum_unrolling<Vec>() == NoUnrolling);
  STATIC_CHECK(sum_unrolling<Mat>() == NoUnrolling);
  // Matters where a block takes DefaultTraversal: SliceVectorizedTraversal ignores Unrolling.
  STATIC_CHECK(sum_unrolling<ConstBlock>() == NoUnrolling);
  // LinearTraversal for complex, whose budget scales with the real packet size although it reads no packets.
  STATIC_CHECK(sum_unrolling<typename Mat::CwiseAbs2ReturnType>() == NoUnrolling);
  STATIC_CHECK(packetwise_sum_unrolling<Panel>() == NoUnrolling);

  // Shifted so that the sums do not cancel, and long enough to read packets of 128 scalars.
  const Vec v = Vec::Constant(300, Scalar(8)) + random_for_arithmetic<Vec>(300);
  const Mat m = Mat::Constant(300, 11, Scalar(8)) + random_for_arithmetic<Mat>(300, 11);
  const ConstBlock block = m.block(1, 2, 15, 7);
  VERIFY_IS_APPROX(v.sum(), sum_by_coeff(v));
  VERIFY_IS_APPROX(m.sum(), sum_by_coeff(m));
  VERIFY_IS_APPROX(block.sum(), sum_by_coeff(block));
  VERIFY_IS_APPROX(v.squaredNorm(), sum_by_coeff(v.cwiseAbs2()));
  VERIFY_IS_APPROX(m.squaredNorm(), sum_by_coeff(m.cwiseAbs2()));
  VERIFY_IS_APPROX(v.cwiseAbs2().sum(), sum_by_coeff(v.cwiseAbs2()));
  VERIFY_IS_APPROX(m.cwiseAbs2().sum(), sum_by_coeff(m.cwiseAbs2()));
  // rowwise() of a column-major operand and colwise() of a row-major one reduce panels of packets.
  VERIFY_IS_APPROX(m.rowwise().sum().sum(), sum_by_coeff(m));
  VERIFY_IS_APPROX(m.transpose().colwise().sum().sum(), sum_by_coeff(m));
}

template <typename Scalar>
void redux_fixed_size() {
  using Vec = Matrix<Scalar, 4, 1>;
  using Panel = Block<const Matrix<Scalar, Dynamic, 4, ColMajor>, 4, 4>;
  STATIC_CHECK(sum_unrolling<Vec>() == CompleteUnrolling);
  STATIC_CHECK(packetwise_sum_unrolling<Panel>() == CompleteUnrolling);
  const Vec v = Vec::Constant(Scalar(8)) + Vec::Random();
  VERIFY_IS_APPROX(v.sum(), sum_by_coeff(v));
}

EIGEN_DECLARE_TEST(redux_unrolling) {
  CALL_SUBTEST(redux_runtime_size<float>());
  CALL_SUBTEST(redux_runtime_size<double>());
  CALL_SUBTEST(redux_runtime_size<long double>());
  CALL_SUBTEST(redux_runtime_size<std::complex<float>>());
  CALL_SUBTEST(redux_runtime_size<std::complex<double>>());
  CALL_SUBTEST(redux_runtime_size<int>());
  CALL_SUBTEST(redux_fixed_size<float>());
  CALL_SUBTEST(redux_fixed_size<std::complex<double>>());
}
