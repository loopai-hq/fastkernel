<!-- Modified by Pulsar. -->
# Contributing

Issues are welcome at [github.com/loopai-hq/pulsar](https://github.com/loopai-hq/pulsar/issues):
bugs, questions about running Pulsar on your Mac, and measurements from Macs we haven't measured. Include the
commit, the Mac (chip, GPU cores, memory), the macOS version and the command you ran.

A change to the engine has to keep the output exact: it should pass `make test-engine-cpu`, `make test-engine-metal`
and the model oracle ([README.md](README.md), "Check the output"). If it claims a speedup, say how it was measured.

Pulsar is built on Splash. [DEVELOPMENT.md](DEVELOPMENT.md), from Splash, covers building from source,
the test suite and model packages.
