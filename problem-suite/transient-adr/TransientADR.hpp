#ifndef TRANSIENT_ADR_HPP
#define TRANSIENT_ADR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class TransientADR
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  using ScalarFunction = std::function<double(const Point<dim> &)>;
  using VectorFunction = std::function<Tensor<1, dim>(const Point<dim> &)>;
  using TimeFunction = std::function<double(const Point<dim> &, double)>;
  using BoundaryConditions = std::map<types::boundary_id, TimeFunction>;

  // Adapter from a lambda to a deal.II Function.
  class FunctionWrapper : public Function<dim>
  {
  public:
    FunctionWrapper(const TimeFunction &function_, const double time_)
      : function(function_)
      , time(time_)
    {}

    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return function(p, time);
    }

  private:
    TimeFunction function;
    double       time;
  };

  // Constructor.
  TransientADR(const std::string &mesh_file_name_,
               const unsigned int r_,
               const double T_,
               const double theta_,
               const double delta_t_,
               const ScalarFunction &mu_,
               const VectorFunction &beta_,
               const ScalarFunction &sigma_,
               const TimeFunction &f_,
               const TimeFunction &initial_condition_,
               const BoundaryConditions &dirichlet_conditions_,
               const BoundaryConditions &neumann_conditions_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , T(T_)
    , theta(theta_)
    , delta_t(delta_t_)
    , mu(mu_)
    , beta(beta_)
    , sigma(sigma_)
    , f(f_)
    , initial_condition(initial_condition_)
    , dirichlet_conditions(dirichlet_conditions_)
    , neumann_conditions(neumann_conditions_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Run the time-dependent simulation.
  void
  run();

protected:
  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve_linear_system();

  // Output.
  void
  output() const;

  // Name of the mesh.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Final time.
  const double T;

  // Theta parameter for the theta method.
  const double theta;

  // Time step.
  const double delta_t;

  // Current time.
  double time = 0.0;

  // Current timestep number.
  unsigned int timestep_number = 0;

  // Diffusion coefficient.
  ScalarFunction mu;

  // Transport velocity.
  VectorFunction beta;

  // Reaction coefficient.
  ScalarFunction sigma;

  // Forcing term.
  TimeFunction f;

  // Initial condition.
  TimeFunction initial_condition;

  // Boundary conditions.
  BoundaryConditions dirichlet_conditions;
  BoundaryConditions neumann_conditions;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // Rank of the current MPI process.
  const unsigned int mpi_rank;

  // Triangulation.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution, without ghost elements.
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution, with ghost elements.
  TrilinosWrappers::MPI::Vector solution;

  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif
