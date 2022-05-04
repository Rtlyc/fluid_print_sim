# Fluid Print Sim

Fluid_print_sim focuses on 2D fluid simulation with [taichi](https://github.com/taichi-dev/taichi) programming language. This includes two fluid simulation examples.

- [Fluid Solid](#fluid-solid)
- [Fluid Print](#fluid-print)

## Fluid Solid

Fluid Solid example shows the interation between fluid and solid. A solid square falls into a water pool.

<img src="pics/fluid_solid_example.gif" alt="Give india logo" width="200"/>

## Fluid Print

Fluid Print example simulates a fluid printer head pouring viscous liquid which is hardening till eventually solid. 

<img src="pics/fluid_print_example.gif" alt="Give india logo" width="200"/>

### Features
- Simulate liquid pouring in real time
- Fluid hardening
- Dynamically control particle interation
- Boundary check
- Fluid turns to solid
- Large memory management

### Usage

```
python fluid_printer.py
```