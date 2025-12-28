# Football Pressure Metric

## URL to Web App / Website
https://pressure-metric-project-qgmw2egnnxs3kz88yk6r2j.streamlit.app/

## Video URL
https://youtu.be/Ptz9yaRjDgQ

---

## Abstract

### Introduction
Pressure is a fundamental concept in football analysis, yet it is most often discussed qualitatively rather than measured in a consistent way. This project introduces a Football Pressure Metric designed to quantify the level of defensive pressure applied to a player in possession. The model combines spatial proximity, defender closing speed, and contextual engagement to produce a continuous, interpretable score that reflects how constrained a player’s decision-making is at any moment. By formalizing pressure in this way, the metric allows for more objective comparison across players, teams, and match phases.

### Use Case(s)
The metric is designed to support analysis at both the event level and the aggregate level. At the event level, it provides frame-by-frame insight into how pressure evolves during individual actions such as carries, receptions, or passes. At the aggregate level, pressure can be summarized over defined time windows (for example, 15-minute intervals, halves, or full matches) to identify broader patterns in how teams apply or absorb pressure.

Key applications include:
- **Pre-match analysis:** profiling how opponents generate pressure and which players or zones are most frequently targeted.  
- **In-match monitoring:** identifying sustained pressure trends that may indicate tactical mismatches or physical fatigue.  
- **Post-match review:** diagnosing moments where pressure contributed to turnovers or rushed decisions, and evaluating alternative tactical responses.

### Potential Audience
This work is intended for performance analysts, coaching staff, and recruitment teams operating across professional, academy, and collegiate football environments. It also has applications in player evaluation, where understanding how individuals perform under varying pressure levels can support scouting and development decisions. More generally, the metric offers a practical way for analysts to incorporate pressure into quantitative football analysis.

---

## Run Instructions

The primary way to interact with this project is through the hosted web app linked above.  
The instructions below are provided for reproducibility and for anyone who wishes to run the app locally.

### Prerequisites
- Python 3.10 or newer  
- `pip` installed  

### Local Setup

Clone the repository:
```bash
git clone https://github.com/rkillip/pressure-metric-project.git
cd pressure-metric-project
