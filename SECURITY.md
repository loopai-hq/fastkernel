<!-- Modified by Pulsar. -->
# Security

Please report vulnerabilities privately to the maintainers of
[github.com/loopai-hq/pulsar](https://github.com/loopai-hq/pulsar) (a GitHub security advisory),
not in a public issue. Include the commit, the Mac and macOS version, and steps to reproduce.

Splash serves on `127.0.0.1`, and authentication is off by default. Before
exposing the server beyond the local machine, set a key with `SPLASH_API_KEY`
or `pulsar serve --api-key`; API requests then need it as a bearer token or
`x-api-key`. `--allowed-origin '*'` lets every web page open in a browser that
reaches the server use it, so set a key with it. Crash traces, which Splash
records only with `SPLASH_CRASH_TRACE=1`, can contain private conversation
data. Exposing the server without a key or a proxy is outside the threat model.

Pulsar is built on Splash; the server and this threat model come from Splash.
