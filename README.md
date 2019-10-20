DEPRECATION NOTICE: I no longer plan to maintain this repository as of October 2019, since matplotlib is not well suited in terms of performance for this project. I have instead moved on to a PyQtGraph based project, [pyqtgraph-spectrographer](https://github.com/flothesof/pyqtgraph-spectrographer).

# LiveFFTPitchTracker

This repository is dedicated to several tools.

A first one is a live FFT plotter, which plots the signal acquired by the microphone and its spectrum.

A second one implements several pitch tracking algorithms to find the pitch of the sound currently being played. In particular, investigated algorithms are:

- harmonic product spectrum
- maximum likelihood 
