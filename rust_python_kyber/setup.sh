#!/bin/sh

Vs="512"
Cwd=$(pwd)
for V in $Vs; do
    echo "Building PQClean for $V.."
    cd ./PQClean/crypto_kem/kyber$V/clean/ && make
    cd ${Cwd}
    cp Cargo$V.toml Cargo.toml
    #echo "Bulding $V.." 
    #maturin build
    echo "Installing $V.."
    maturin develop --release
done
