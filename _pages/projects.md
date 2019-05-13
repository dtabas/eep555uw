---
permalink: /projects/
title: "Current and past projects"
excerpt: "This page contains descriptions of current and past projects"
author_profile: true
---

## Current Projects

### Coincident peak timing in electrical markets

Coincident peaks are a pricing mechanism in electrical markets based on the timing of the peak load across the _entire_ system, not just individual consumers. We show that the timing of these peaks can be [predicted](https://ieeexplore.ieee.org/abstract/document/8646654) with a purposefully simple neural network, implying that complex market behaviors involving hedging may emerge for large consumers with demand flexibility. For small consumers, we derive an optimal strategy using dynamic programming. For large consumers, their strategies more dramatically interact and we are investigating the existence of Nash equillibria.

<p align="center">
  <img width="837" height="466" src="/images/CP_diagram.png" alt="Coincident peak data model">
</p>

---
### Transfer learning for linear state estimators of commercial HVAC systems

Learning for efficient state estimation and control of energy usage commercial HVAC systems is stymied by a lack of labeled data. The state of HVAC systems in commerical buildings can be reasonably approximated by a polynomial time dynamic equation; for a building with abundant sensor data we learn an accurate model as a base for transfering to buildings in which sensor deployment is limited. 

======

## Past Projects

### Measuring congestion due to drivers cruising for parking

We estimate the proportion of drivers cruising for parking from open-source paid-parking transaction data using [queueing network](https://ieeexplore.ieee.org/abstract/document/8663628) techniques borrowed from communications theory. Combined with Google maps travel time estimates, costs to social welfare can also be estimated. With these results, [optimization](https://ieeexplore.ieee.org/abstract/document/8264412) of curbside parking resources with respect to congestion constraints can be targeted to [high demand regions](https://ieeexplore.ieee.org/abstract/document/8431681) where congestion resulting from cruising originates.

<p align="center">
  <img width="806" height="382" src="/images/blockface.png" alt="curbside parking model">
</p>
