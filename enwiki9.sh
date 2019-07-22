#!/usr/bin/env bash

# see http://mattmahoney.net/dc/textdata.html
wget --continue http://mattmahoney.net/dc/enwik9.zip

unzip -p enwik9.zip enwik9 | perl text8_conversion.pl | head -c 100000000 > data/text8

ls -lh