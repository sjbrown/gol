#! /usr/bin/env bash

mkdir -p build
rm -rf build/gol.zip
rm build/*.py
cp next_iteration.py build/
cp rules_cache.py build/
cp lambda_handler.py build/
touch build/__init__.py
cd build
git clone https://github.com/vitolimandibhrata/aws-lambda-numpy.git
mv aws-lambda-numpy/lib ./lib
mv aws-lambda-numpy/numpy ./numpy
rm -rf aws-lambda-numpy
zip -r gol .
