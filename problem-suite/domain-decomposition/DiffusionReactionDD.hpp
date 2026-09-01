#ifndef DIFFUSION_REACTION_DD_HPP
#define DIFFUSION_REACTION_DD_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using namespace dealii;

class DiffusionReactionDD
{
public:
  static constexpr unsigned int dim = 2;

  using ScalarFunction = std::function<double(const Point<dim> &)>;
  using DirichletConditions =
    std::map<types::boundary_id, const Function<dim> *>;
  using NeumannConditions =
    std::map<types::boundary_id, ScalarFunction>;

  DiffusionReactionDD(const unsigned int subdomain_id_,
                      const std::string &mesh_file_name_,
                      const unsigned int degree_,
                      const types::boundary_id interface_id_,
                      const ScalarFunction &alpha_,
                      const ScalarFunction &gamma_,
                      const ScalarFunction &f_,
                      const DirichletConditions &dirichlet_conditions_,
                      const NeumannConditions &neumann_conditions_)
    : subdomain_id(subdomain_id_)
    , mesh_file_name(mesh_file_name_)
    , degree(degree_)
    , interface_id(interface_id_)
    , alpha(alpha_)
    , gamma(gamma_)
    , f(f_)
    , dirichlet_conditions(dirichlet_conditions_)
    , neumann_conditions(neumann_conditions_)
  {}

  void setup();
  void assemble();
  void solve();
  void output(const unsigned int iteration) const;

  void apply_interface_dirichlet(const DiffusionReactionDD &other);
  double apply_interface_correction(DiffusionReactionDD &other,
                                    const double lambda,
                                    const double smoothing);

protected:
  std::map<types::global_dof_index, types::global_dof_index>
  compute_interface_map(const DiffusionReactionDD &other) const;

  Vector<double> compute_interface_residual();

  const unsigned int subdomain_id;
  const std::string mesh_file_name;
  const unsigned int degree;
  const types::boundary_id interface_id;

  ScalarFunction alpha;
  ScalarFunction gamma;
  ScalarFunction f;
  DirichletConditions dirichlet_conditions;
  NeumannConditions neumann_conditions;

  Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;
  DoFHandler<dim> dof_handler;
  std::map<types::global_dof_index, Point<dim>> support_points;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;
  Vector<double> solution;
};

#endif
