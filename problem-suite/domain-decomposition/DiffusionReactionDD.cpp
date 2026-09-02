#include "DiffusionReactionDD.hpp"

void DiffusionReactionDD::setup()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(mesh);
  std::ifstream mesh_file(mesh_file_name);
  grid_in.read_msh(mesh_file);

  fe                  = std::make_unique<FE_SimplexP<dim>>(degree);
  quadrature          = std::make_unique<QGaussSimplex<dim>>(degree + 1);
  quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(degree + 1);

  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);

  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);
  support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  system_rhs.reinit(dof_handler.n_dofs());
  solution.reinit(dof_handler.n_dofs());
}

void DiffusionReactionDD::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  FEValues<dim>     fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe, *quadrature_boundary, update_values | update_quadrature_points | update_JxW_values);

  FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                       cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < quadrature->size(); ++q)
        {
          const Point<dim> &point     = fe_values.quadrature_point(q);
          const double      alpha_loc = alpha(point);
          const double      gamma_loc = gamma(point);
          const double      f_loc     = f(point);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += (alpha_loc * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                                      gamma_loc * fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) *
                                     fe_values.JxW(q);

              cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      if (cell->at_boundary())
        for (unsigned int face = 0; face < cell->n_faces(); ++face)
          if (cell->face(face)->at_boundary())
            {
              const auto condition = neumann_conditions.find(cell->face(face)->boundary_id());
              if (condition == neumann_conditions.end())
                continue;

              fe_face_values.reinit(cell, face);
              for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) += condition->second(fe_face_values.quadrature_point(q)) * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
            }

      cell->get_dof_indices(dof_indices);
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, dirichlet_conditions, boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);
}

void DiffusionReactionDD::solve()
{
  SolverControl                          solver_control(1000, std::max(1e-16, 1e-12 * system_rhs.l2_norm()));
  SolverCG<Vector<double>>               solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
}

void DiffusionReactionDD::output(const unsigned int iteration) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  const std::string file_name = "output-dd-" + std::to_string(subdomain_id) + "-" + std::to_string(iteration) + ".vtk";
  std::ofstream     output_file(file_name);
  data_out.write_vtk(output_file);
}

void DiffusionReactionDD::apply_interface_dirichlet(const DiffusionReactionDD &other)
{
  const auto                                interface_map = compute_interface_map(other);
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &[dof, other_dof] : interface_map)
    boundary_values[dof] = other.solution[other_dof];

  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);
}

double DiffusionReactionDD::apply_interface_correction(DiffusionReactionDD &other, const double lambda, const double smoothing)
{
  const auto           interface_map  = compute_interface_map(other);
  const Vector<double> residual       = compute_interface_residual();
  const Vector<double> other_residual = other.compute_interface_residual();

  std::vector<types::global_dof_index>            interface_dofs;
  std::map<types::global_dof_index, unsigned int> interface_indices;
  for (const auto &[dof, other_dof] : interface_map)
    {
      interface_indices[dof] = interface_dofs.size();
      interface_dofs.push_back(dof);
    }

  const QGaussSimplex<dim - 1> quadrature_interface(degree + 1);
  FEFaceValues<dim> fe_face_values(*fe, quadrature_interface, update_values | update_gradients | update_normal_vectors | update_JxW_values);

  FullMatrix<double>                   interface_matrix(interface_dofs.size(), interface_dofs.size());
  std::vector<types::global_dof_index> dof_indices(fe->dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->at_boundary())
      for (unsigned int face = 0; face < cell->n_faces(); ++face)
        if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == interface_id)
          {
            fe_face_values.reinit(cell, face);
            cell->get_dof_indices(dof_indices);

            for (unsigned int q = 0; q < quadrature_interface.size(); ++q)
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                  if (interface_indices.count(dof_indices[i]) > 0 && interface_indices.count(dof_indices[j]) > 0)
                    {
                      const Tensor<1, dim> normal             = fe_face_values.normal_vector(q);
                      const Tensor<1, dim> grad_i             = fe_face_values.shape_grad(i, q);
                      const Tensor<1, dim> grad_j             = fe_face_values.shape_grad(j, q);
                      const double         tangential_product = grad_i * grad_j - (grad_i * normal) * (grad_j * normal);

                      interface_matrix(interface_indices.at(dof_indices[i]), interface_indices.at(dof_indices[j])) +=
                        (fe_face_values.shape_value(i, q) * fe_face_values.shape_value(j, q) + smoothing * tangential_product) *
                        fe_face_values.JxW(q);
                    }
          }

  Vector<double> flux_imbalance(interface_dofs.size());
  for (const auto &[dof, other_dof] : interface_map)
    flux_imbalance[interface_indices.at(dof)] = residual[dof] + other_residual[other_dof];

  Vector<double> correction(interface_dofs.size());
  interface_matrix.gauss_jordan();
  interface_matrix.vmult(correction, flux_imbalance);
  correction *= lambda;

  for (const auto &[dof, other_dof] : interface_map)
    {
      solution[dof] -= correction[interface_indices.at(dof)];
      other.solution[other_dof] = solution[dof];
    }

  return correction.l2_norm();
}

Vector<double> DiffusionReactionDD::compute_interface_residual()
{
  assemble();
  Vector<double> residual = system_rhs;
  residual *= -1.0;
  system_matrix.vmult_add(residual, solution);
  return residual;
}

std::map<types::global_dof_index, types::global_dof_index> DiffusionReactionDD::compute_interface_map(const DiffusionReactionDD &other) const
{
  const IndexSet current_interface = DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {interface_id});
  const IndexSet other_interface   = DoFTools::extract_boundary_dofs(other.dof_handler, ComponentMask(), {other.interface_id});

  std::set<types::boundary_id> current_dirichlet_ids;
  for (const auto &[boundary_id, function] : dirichlet_conditions)
    current_dirichlet_ids.insert(boundary_id);
  IndexSet current_dirichlet(dof_handler.n_dofs());
  if (!current_dirichlet_ids.empty())
    current_dirichlet = DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), current_dirichlet_ids);

  std::set<types::boundary_id> other_dirichlet_ids;
  for (const auto &[boundary_id, function] : other.dirichlet_conditions)
    other_dirichlet_ids.insert(boundary_id);
  IndexSet other_dirichlet(other.dof_handler.n_dofs());
  if (!other_dirichlet_ids.empty())
    other_dirichlet = DoFTools::extract_boundary_dofs(other.dof_handler, ComponentMask(), other_dirichlet_ids);

  std::map<types::global_dof_index, types::global_dof_index> interface_map;
  for (const auto dof : current_interface)
    {
      if (current_dirichlet.is_element(dof))
        continue;

      const Point<dim>       &point            = support_points.at(dof);
      types::global_dof_index nearest          = numbers::invalid_dof_index;
      double                  nearest_distance = std::numeric_limits<double>::max();

      for (const auto other_dof : other_interface)
        if (!other_dirichlet.is_element(other_dof))
          {
            const double distance = point.distance_square(other.support_points.at(other_dof));
            if (distance < nearest_distance)
              {
                nearest          = other_dof;
                nearest_distance = distance;
              }
          }

      AssertThrow(nearest != numbers::invalid_dof_index, ExcMessage("No matching interface DoF found."));
      interface_map[dof] = nearest;
    }

  return interface_map;
}
