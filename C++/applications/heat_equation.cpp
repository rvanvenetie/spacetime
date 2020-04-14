#include "heat_equation.hpp"
namespace applications {
template <>
void HeatEquation<true>::InitializeBT() {
  BT_ = std::make_shared<TypeBT>(B_->Transpose(), vec_Y_in_.get(),
                                 vec_X_out_.get());
}

template <>
void HeatEquation<false>::InitializeBT() {
  auto BT_t = std::make_shared<
      BilinearForm<Time::TransportOperator, space::MassOperator,
                   OrthonormalWaveletFn, ThreePointWaveletFn>>(
      vec_Y_in_.get(), vec_X_out_.get(), B_->theta(), B_->sigma());
  auto BT_s = std::make_shared<
      BilinearForm<Time::MassOperator, space::StiffnessOperator,
                   OrthonormalWaveletFn, ThreePointWaveletFn>>(
      vec_Y_in_.get(), vec_X_out_.get(), B_->theta(), B_->sigma());
  BT_ = std::make_shared<TypeBT>(BT_t, BT_s);
}
};  // namespace applications
