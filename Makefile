# The makefile for mav_gesture_control
PROJECT := mav_gesture_control

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
RELEASE_BUILD_DIR := .$(BUILD_DIR)_release
DEBUG_BUILD_DIR := .$(BUILD_DIR)_debug

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
endif

# The target shared library and static library name
LIB_BUILD_DIR := $(BUILD_DIR)/lib
NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).a

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src -name "*.cpp")
# HXX_SRCS are the header files
HXX_SRCS := $(shell find include -name "*.h")
# TOOL_SRCS are the source files for the tool binaries
TOOL_SRCS := $(shell find tools -name "*.cpp")
# EXAMPLE_SRCS are the source files for the example binaries
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src

##############################
# Derive generated files
##############################
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the tool, example, and test objects.
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
OBJ_BUILD_DIR := $(BUILD_DIR)/src
AR_BUILD_DIR := $(OBJ_BUILD_DIR)/action_recognition
CVBLOB_BUILD_DIR := $(OBJ_BUILD_DIR)/cvblob
DETECTION_BUILD_DIR := $(OBJ_BUILD_DIR)/detection
IDT_BUILD_DIR := $(OBJ_BUILD_DIR)/improved_trajectories
VM_BUILD_DIR := $(OBJ_BUILD_DIR)/visual_mapping
UTIL_BUILD_DIR := $(OBJ_BUILD_DIR)/utils
OBJS := $(CXX_OBJS)
# tool, example, and test objects
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
TOOL_BUILD_DIR := $(BUILD_DIR)/tools
MODELS_LINK = $(TOOL_BUILD_DIR)/models
# tool, example, and test bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
# symlinks to tool bins without the ".bin" extension
TOOL_BIN_LINKS := ${TOOL_BINS:.bin=}

##############################
# Derive compiler warning dump locations
##############################
WARNS_EXT := warnings.txt
CXX_WARNS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o.$(WARNS_EXT)})
CU_WARNS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.cuo.$(WARNS_EXT)})
ALL_CXX_WARNS := $(CXX_WARNS) $(TOOL_WARNS) $(EXAMPLE_WARNS) $(TEST_WARNS)
ALL_WARNS := $(ALL_CXX_WARNS) $(ALL_CU_WARNS)

EMPTY_WARN_REPORT := $(BUILD_DIR)/.$(WARNS_EXT)
NONEMPTY_WARN_REPORT := $(BUILD_DIR)/$(WARNS_EXT)

##############################
# Derive include and lib directories
##############################
INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ./src \
	$(shell \
	find ./include -type d)
LIBRARIES += boost_system boost_filesystem boost_regex boost_serialization \
	opencv_calib3d opencv_contrib opencv_core opencv_features2d opencv_gpu opencv_flann opencv_highgui \
	opencv_imgproc opencv_legacy opencv_ml opencv_nonfree opencv_objdetect opencv_ocl opencv_photo \
	opencv_stitching opencv_superres opencv_ts opencv_video opencv_videostab \
	pthread
	

WARNINGS := -Wall -Wno-sign-compare

##############################
# Set build directories
##############################
ALL_BUILD_DIRS := $(sort \
		$(BUILD_DIR) $(LIB_BUILD_DIR) $(OBJ_BUILD_DIR) \
		$(AR_BUILD_DIR) $(CVBLOB_BUILD_DIR) $(DETECTION_BUILD_DIR) \
		$(IDT_BUILD_DIR) $(VM_BUILD_DIR) $(UTIL_BUILD_DIR) $(TOOL_BUILD_DIR))

##############################
# Set directory for Doxygen-generated documentation
##############################
DOXYGEN_CONFIG_FILE ?= ./.Doxyfile
# should be the same as OUTPUT_DIRECTORY in the .Doxyfile
DOXYGEN_OUTPUT_DIR ?= ./doxygen
DOXYGEN_COMMAND ?= doxygen
# All the files that might have Doxygen documentation.
DOXYGEN_SOURCES := $(shell find \
	src \
	include \
	tools \
	-name "*.cpp" -or -name "*.h")
DOXYGEN_SOURCES += $(DOXYGEN_CONFIG_FILE)


##############################
# Configure build
##############################

CXX ?= /usr/bin/g++
GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
# older versions of gcc are too dumb to build boost with -Wuninitalized
ifeq ($(shell echo $(GCCVERSION) \< 4.6 | bc), 1)
	WARNINGS += -Wno-uninitialized
endif
# boost::thread is reasonably called boost_thread (compare OS X)
LIBRARIES += boost_thread


# Custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
else
	COMMON_FLAGS += -DNDEBUG -O2
endif

# CPU-only configuration
ifeq ($(CPU_ONLY), 0)
	COMMON_FLAGS += -D CUDA
endif

OBJS := $(CXX_OBJS)
ALL_WARNS := $(ALL_CXX_WARNS)



# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
LINKFLAGS += -fPIC $(COMMON_FLAGS) $(WARNINGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

# 'superclean' target recursively* deletes all files ending with an extension
# in $(SUPERCLEAN_EXTS) below. 
#
# 'supercleanlist' will list the files to be deleted by make superclean.
#
# * Recursive with the exception that symbolic links are never followed, per the
# default behavior of 'find'.
SUPERCLEAN_EXTS := .so .a .o .bin .testbin .pb.cc .pb.h _pb2.py .cuo

##############################
# Define build targets
##############################
.PHONY: all clean docs linecount tools 
	superclean supercleanlist supercleanfiles warn everything

all: $(NAME) $(STATIC_NAME) tools



linecount:
	cloc --read-lang-def=$(PROJECT).cloc \
		src include tools examples \
		python matlab


docs: $(DOXYGEN_OUTPUT_DIR)
	@ mkdir -p ./docs
	@ ln -sfn ../$(DOXYGEN_OUTPUT_DIR)/html ./docs/doxygen

$(DOXYGEN_OUTPUT_DIR): $(DOXYGEN_CONFIG_FILE) $(DOXYGEN_SOURCES)
	$(DOXYGEN_COMMAND) $(DOXYGEN_CONFIG_FILE)


tools: $(TOOL_BINS) $(TOOL_BIN_LINKS) $(MODELS_LINK)


warn: $(EMPTY_WARN_REPORT)

$(EMPTY_WARN_REPORT): $(ALL_WARNS) | $(BUILD_DIR)
	@ cat $(ALL_WARNS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_WARN_REPORT); \
		echo "Compiler produced one or more warnings."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_WARN_REPORT); \
	  echo "No compiler warnings!";

$(ALL_CXX_WARNS): %.o.$(WARNS_EXT) : %.o

$(ALL_CU_WARNS): %.cuo.$(WARNS_EXT) : %.cuo

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

# Create a target ".linked" in this BUILD_DIR to tell Make that the "build" link
# is currently correct, then delete the one in the OTHER_BUILD_DIR in case it
# exists and $(DEBUG) is toggled later.
$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) $(OTHER_BUILD_DIR)/.linked
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)	
	@ touch $@
	@ echo

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@	
	@ echo
	
$(MODELS_LINK):	| $(TOOL_BUILD_DIR)
	@ echo ln -s $(abspath $(BUILD_DIR))/../models $@
	@ ln -s $(abspath $(BUILD_DIR))/../models $@


$(NAME): $(OBJS) | $(LIB_BUILD_DIR)
	$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	ar rcs $@ $(OBJS)
	@ echo

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@
	

$(TOOL_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(AR_BUILD_DIR)/%.o: src/action_recognition/%.cpp $(HXX_SRCS) \
		| $(AR_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo
	
$(CVBLOB_BUILD_DIR)/%.o: src/cvblob/%.cpp $(HXX_SRCS) \
		| $(CVBLOB_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(DETECTION_BUILD_DIR)/%.o: src/detection/%.cpp $(HXX_SRCS) \
		| $(DETECTION_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(IDT_BUILD_DIR)/%.o: src/improved_trajectories/%.cpp $(HXX_SRCS) \
		| $(IDT_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(VM_BUILD_DIR)/%.o: src/visual_mapping/%.cpp $(HXX_SRCS) \
		| $(VM_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo
	
$(UTIL_BUILD_DIR)/%.o: src/utils/%.cpp $(HXX_SRCS) | $(UTIL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo


$(TOOL_BUILD_DIR)/%.o: tools/%.cpp $(PROTO_GEN_HEADER) | $(TOOL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(BUILD_DIR)/src/%.o: src/%.cpp $(HXX_SRCS)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo


clean:
	@- $(RM) -rf $(ALL_BUILD_DIRS)
	@- $(RM) -rf $(OTHER_BUILD_DIR)
	@- $(RM) -rf $(BUILD_DIR_LINK)
	@- $(RM) -rf $(DISTRIBUTE_DIR)
	@- $(RM) $(PY$(PROJECT)_SO)
	@- $(RM) $(MAT$(PROJECT)_SO)

supercleanfiles:
	$(eval SUPERCLEAN_FILES := $(strip \
			$(foreach ext,$(SUPERCLEAN_EXTS), $(shell find . -name '*$(ext)' \
			-not -path './data/*'))))

supercleanlist: supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
	fi

superclean: clean supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo "Deleting the following generated files:"; \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
		$(RM) $(SUPERCLEAN_FILES); \
	fi

