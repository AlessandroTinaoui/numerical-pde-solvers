#include "TransientStokes.hpp"

static constexpr unsigned int dim = 3;

// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/mesh-step-5.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  // Time discretization.
  const double final_time = 1.0;
  const double delta_t    = 0.1;
  const double theta      = 1.0; // Backward Euler.

  // Problem coefficients.
  const auto mu                = [](const Point<dim> &) { return 1.0; };
  const auto alpha             = [](const Point<dim> &) { return 0.0; };
  const auto forcing           = [](const Point<dim> &, const double) { return Tensor<1, dim>(); };
  const auto initial_condition = [](const Point<dim> &, const double) { return Tensor<1, dim>(); };

  // Boundary conditions.
  TransientStokes<dim>::DirichletConditions dirichlet;
  TransientStokes<dim>::TractionConditions  traction;

  dirichlet[1] = [](const Point<dim> &, const double) { return Tensor<1, dim>(); };

  const double p_in  = 10.0;
  const double p_out = 0.0;

  traction[0] = [p_in](const Point<dim> &, const Tensor<1, dim> &normal, const double) { return -p_in * normal; };
  traction[2] = [p_out](const Point<dim> &, const Tensor<1, dim> &normal, const double) { return -p_out * normal; };

  const bool fix_pressure_nullspace = false;

  TransientStokes<dim> problem(mesh_file_name,
                               degree_velocity,
                               degree_pressure,
                               final_time,
                               delta_t,
                               mu,
                               alpha,
                               forcing,
                               initial_condition,
                               dirichlet,
                               traction,
                               theta,
                               fix_pressure_nullspace);

  problem.run();

  return 0;
}
