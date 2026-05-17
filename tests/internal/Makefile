CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
INCLUDES  = -I../../csrc -I.

TESTS = test_progress

CSRC = ../../csrc

.PHONY: all run clean
all: $(TESTS)

run: $(TESTS)
	@fails=0; for t in $(TESTS); do \
	  echo "==== $$t ===="; \
	  ./$$t || fails=$$((fails + 1)); \
	done; \
	exit $$fails

test_progress: test_progress.cpp $(CSRC)/backend/Progress.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(TESTS)
