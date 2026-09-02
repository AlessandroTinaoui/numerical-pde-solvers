#include "SteadyStokes.hpp"

static constexpr unsigned int dim = 3;

// Parabolic inlet profile for the step mesh.
class InletVelocity : public Function<dim>
{
public:
  InletVelocity()
    : Function<dim>(dim + 1)
  {}

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    if (component == 0)
      return -p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);

    return 0.0;
  }
};

// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/mesh-step-5.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  // Problem coefficients.
  const auto mu      = [](const Point<dim> &) { return 1.0; };
  const auto alpha   = [](const Point<dim> &) { return 0.0; };
  const auto forcing = [](const Point<dim> &) { return Tensor<1, dim>(); };

  // Boundary conditions.
  InletVelocity                inlet_velocity;
  Functions::ZeroFunction<dim> zero_velocity(dim + 1);

  SteadyStokes<dim>::DirichletConditions dirichlet;
  SteadyStokes<dim>::TractionConditions  traction;

  dirichlet[0] = &inlet_velocity;
  dirichlet[1] = &zero_velocity;

  const double p_out = 10.0;
  traction[2]        = [p_out](const Point<dim> &, const Tensor<1, dim> &normal) { return -p_out * normal; };

  const bool fix_pressure_nullspace = false;

  SteadyStokes<dim> problem(mesh_file_name, degree_velocity, degree_pressure, mu, alpha, forcing, dirichlet, traction, fix_pressure_nullspace);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
