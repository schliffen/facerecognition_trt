# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ekin/frproj/retinaface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ekin/frproj/retinaface/build

# Include any dependencies generated for this target.
include CMakeFiles/decodeplugin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/decodeplugin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/decodeplugin.dir/flags.make

CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o: CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o.depend
CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o: CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o.Debug.cmake
CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o: ../decode.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ekin/frproj/retinaface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o"
	cd /home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir && /usr/local/bin/cmake -E make_directory /home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir//.
	cd /home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir//./decodeplugin_generated_decode.cu.o -D generated_cubin_file:STRING=/home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir//./decodeplugin_generated_decode.cu.o.cubin.txt -P /home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir//decodeplugin_generated_decode.cu.o.Debug.cmake

# Object files for target decodeplugin
decodeplugin_OBJECTS =

# External object files for target decodeplugin
decodeplugin_EXTERNAL_OBJECTS = \
"/home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o"

libdecodeplugin.so: CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o
libdecodeplugin.so: CMakeFiles/decodeplugin.dir/build.make
libdecodeplugin.so: /usr/local/cuda-10.2/lib64/libcudart.so
libdecodeplugin.so: CMakeFiles/decodeplugin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ekin/frproj/retinaface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libdecodeplugin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/decodeplugin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/decodeplugin.dir/build: libdecodeplugin.so

.PHONY : CMakeFiles/decodeplugin.dir/build

CMakeFiles/decodeplugin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/decodeplugin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/decodeplugin.dir/clean

CMakeFiles/decodeplugin.dir/depend: CMakeFiles/decodeplugin.dir/decodeplugin_generated_decode.cu.o
	cd /home/ekin/frproj/retinaface/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ekin/frproj/retinaface /home/ekin/frproj/retinaface /home/ekin/frproj/retinaface/build /home/ekin/frproj/retinaface/build /home/ekin/frproj/retinaface/build/CMakeFiles/decodeplugin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/decodeplugin.dir/depend

