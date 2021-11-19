---
permalink: /projects/
title: ""
excerpt: "This page contains descriptions of current and past projects"
author_profile: true
---

## Current Projects

### Dynamic Curb Resource Allocation

Curb space can be viewed as the interfacing layer between a city's surface transportation network and the point of arrival or departure for goods and people. As the variety of mobility services increases, cities are pressed for effective policies. With funding from the Department of Energy's Vehicle Technology Office, PNNL is leading the development of curb resource management and data analysis tools to simulate prospective policies and measure its impact, for example, by computing a curb policy dependent-fundamental diagram on adjacent roadways. We've partnered with the University of Washington [Urban Freight Lab](http://depts.washington.edu/sctlctr/urban-freight-lab-0) to simulate curb activity on a block-by-block scale, Lawrence Berkeley National Laboratory's developers of the [BEAM](https://beam.lbl.gov/) (a city-scale traffic simulator) to measure region-wide energy efficiency and productivity impacts, the National Renewable Energy Laboratory to draw on energy efficiency performance metric expertise, and smart cities tech company [Lacuna](https://lacuna.ai/) to develop state of the art management strategies based on new data streams and standards.

<p align="center">
  <img width="800" height="252" src="/images/Dynamic_Curbs_v4-01.png" alt="Curb demand profiles by modality">
</p>

======

### Coincident peak timing in electrical markets

Coincident peaks are a pricing mechanism in electrical markets based on the timing of the peak load across the _entire_ system, not just individual consumers. We show that the timing of these peaks can be [predicted](https://ieeexplore.ieee.org/abstract/document/8646654) with a purposefully simple neural network, implying that complex market behaviors involving hedging may emerge for large consumers with demand flexibility. For small consumers, we derive an optimal strategy using [dynamic programming](https://arxiv.org/abs/1908.00685). For large consumers, their strategies more dramatically interact and we are investigating the existence of Nash equillibria.

<p align="center">
  <img width="837" height="466" src="/images/CP_diagram.png" alt="Coincident peak data model">
</p>

======

## Past Projects

### Transfer learning for linear state estimators of commercial HVAC systems

Learning for efficient state estimation and control of energy usage commercial HVAC systems is stymied by a lack of labeled data. The state of HVAC systems in commerical buildings can be reasonably approximated by a polynomial time dynamic equation; for a building with abundant sensor data we learn an accurate model as a base for transfering to buildings in which sensor deployment is limited. Between similar buildings, we demonstrate a method of transfering such a model when used for [fault detection](https://arxiv.org/abs/2002.01060).

---

### Measuring congestion due to drivers cruising for parking

We estimate the proportion of drivers cruising for parking from open-source paid-parking [transaction data](https://cpatdowling.github.io/notebooks/demandviz) using [queueing network](https://ieeexplore.ieee.org/abstract/document/8663628) techniques borrowed from communications theory. Combined with Google maps travel time estimates, costs to social welfare can also be estimated. With these results, [optimization](https://ieeexplore.ieee.org/abstract/document/8264412) of curbside parking resources with respect to congestion constraints can be targeted to [high demand regions](https://ieeexplore.ieee.org/abstract/document/8431681) where congestion resulting from cruising originates.

<p align="center">
  <img width="806" height="382" src="/images/blockface.png" alt="curbside parking model">
</p>
