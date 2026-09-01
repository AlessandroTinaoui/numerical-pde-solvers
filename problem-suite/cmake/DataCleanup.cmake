if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR is required")
endif()

file(GLOB generated_data
  "${OUTPUT_DIR}/output-*.vtk"
  "${OUTPUT_DIR}/output-*.vtu"
  "${OUTPUT_DIR}/output-*.pvtu"
  "${OUTPUT_DIR}/output-*.pvd"
  "${OUTPUT_DIR}/output-*.visit"
  "${OUTPUT_DIR}/convergence.csv")

list(LENGTH generated_data generated_data_count)

if(generated_data)
  file(REMOVE ${generated_data})
endif()

message(STATUS "Removed ${generated_data_count} generated data files from ${OUTPUT_DIR}")
