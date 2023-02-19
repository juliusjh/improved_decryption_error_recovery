# Code for "Belief Propagation Meets Lattice Reduction: Security Estimates for Error-Tolerant Key Recovery from Decryption Errors"

To install ensure that recent rust and python3 versions are installed.

Then use "source setup.sh" to create the virtual environment (virtualenv needs to be installed), source it, compile the modified PQClean and the rust packages, and install python packages.

To start a recovery then use "python3 recover.py new 10000" (with no ciphertext filtering and 10000 newly sampled inequalities).

More options are available and are defined in recover.py (or by using recovery --help).

For example, to recover in a filtered scenario as in the paper you may use "python3 recover.py new --max-delta-v 10 6000"

Note that some options which are not described in the paper are only partially implemented or are not tested.

In case of bugs or question, please write an email to the first author of the paper (email in the pdf).
