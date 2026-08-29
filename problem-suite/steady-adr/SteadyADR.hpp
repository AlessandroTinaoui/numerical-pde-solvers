#ifndef STEADY_ADR_HPP
#define STEADY_ADR_HPP

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
 * Class managing the SteadyADR problem in dimension dim.
 */
template <unsigned int dim>
class SteadyADR
{
public:
  using ScalarFunction = std::function<double(const Point<dim> &)>;
  using VectorFunction = std::function<Tensor<1, dim>(const Point<dim> &)>;

  using DirichletConditions = std::map<types::boundary_id, const Function<dim> *>;

  using NeumannConditions = std::map<types::boundary_id, ScalarFunction>;

  // Constructor for a mesh read from a .msh file.
  SteadyADR(const std::string         &mesh_file_name_,
          const unsigned int        &r_,
          const ScalarFunction      &mu_,
          const VectorFunction      &beta_,
          const ScalarFunction      &sigma_,
          const ScalarFunction      &f_,
          const DirichletConditions &dirichlet_conditions_,
          const NeumannConditions   &neumann_conditions_,
          const bool                 fix_nullspace_ = true)
    : mesh_file_name(mesh_file_name_)
    , N_el(0)
    , generate_mesh_1d(false)
    , r(r_)
    , mu(mu_)
    , beta(beta_)
    , sigma(sigma_)
    , f(f_)
    , dirichlet_conditions(dirichlet_conditions_)
    , neumann_conditions(neumann_conditions_)
    , fix_nullspace(fix_nullspace_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Constructor for an automatically generated mesh of the unit interval.
  // This constructor is used for SteadyADR<1>.
  SteadyADR(const unsigned int        &N_el_,
          const unsigned int        &r_,
          const ScalarFunction      &mu_,
          const VectorFunction      &beta_,
          const ScalarFunction      &sigma_,
          const ScalarFunction      &f_,
          const DirichletConditions &dirichlet_conditions_,
          const NeumannConditions   &neumann_conditions_,
          const bool                 fix_nullspace_ = true)
    : N_el(N_el_)
    , generate_mesh_1d(true)
    , r(r_)
    , mu(mu_)
    , beta(beta_)
    , sigma(sigma_)
    , f(f_)
    , dirichlet_conditions(dirichlet_conditions_)
    , neumann_conditions(neumann_conditions_)
    , fix_nullspace(fix_nullspace_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

protected:
  // Name of the mesh file. It is empty for automatically generated meshes.
  const std::string mesh_file_name;

  // Number of elements for the automatically generated 1D mesh.
  const unsigned int N_el;

  // Whether the 1D mesh must be generated automatically.
  const bool generate_mesh_1d;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  ScalarFunction mu;

  // Transport velocity.
  VectorFunction beta;

  // Reaction coefficient.
  ScalarFunction sigma;

  // Forcing term.
  ScalarFunction f;

  // Boundary conditions.
  DirichletConditions dirichlet_conditions;
  NeumannConditions   neumann_conditions;

  // Pin the first DoF for pure-Neumann problems with a constant nullspace.
  const bool fix_nullspace;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // Rank of the current MPI process.
  const unsigned int mpi_rank;

  // Distributed triangulation.
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

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Output stream for process 0.
  ConditionalOStream pcout;

  // Locally owned DoFs for the current process.
  IndexSet locally_owned_dofs;
};

#endif
