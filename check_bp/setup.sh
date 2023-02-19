#!/bin/sh

Vs="512"
Cwd=$(pwd)
for V in $Vs; do
    echo "Installing $V.."
    cp Cargo$V.toml Cargo.toml
    #echo "Bulding $V.." 
    #maturin build
    maturin develop --release
done

rm Cargo.toml

