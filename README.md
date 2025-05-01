# TDACC (Trade Dangerous Accelerator)

## Introduction
Elite Dangerous is an online multiplayer space video game where commanders (players) pilot space ships around a digital recreation of the Milky Way galaxy. One of the activities players can perform in this game earn in-game currency is trading, which is outlined below.

### Trading
The idea of trading in Elite is fairly simple: buy commodities in one system for a low price and transport and sell them in another solar system to make a profit. Naturally, Traders want to maximize their profit. As a result, tools that compute optimal trading routes with market data and various ship and location parameters as an input have been created.

### Current solutions
The two main accepted tools are
* [inara.cz](https://inara.cz/elite/market-traderoutes/)
* [Trade Dangerous](https://github.com/eyeonus/Trade-Dangerous)

These two tools are distinct in their approach. Inara.cz provides *globally* optimal trade routes that it precomputes every so often without much regard for the exact location of the player. On the other hand, Trade Dangerous computes the optimal trade route from the user's starting point. Since Inara's tooling is closed-source (the website simply displays results), from here we shift our focus to Trade Dangerous.

### Motivation
The current problem with Trade Dangerous is that it is incredibly slow (see statistics in following sections). Without a deep dive into the source code, it seems the performance problems can be attributed to the fact the tool is written in python and runs on a single core. From [previous attempts](https://github.com/eyeonus/Trade-Dangerous/pull/197), spreading work across multiple processes/threads is somewhat beneficial. Thus, in this project, I have created a proof of concept that uses industry-standard parallelization techniques and to elicit performance gains.

## Methods
### Algorithm
A Trade route is split into hops and jumps. A jump is a single traversal between systems. A hop is consisted of multiple jumps where commodities are purchased at the beginning of the hop and sold at the end. Another degree of complexity to this problem is that there are generally multiple stations per system where commodities can be bought and sold. Thus, our algorithm must select the most optimal starting and ending stations for each hop.

When researching this algorithm, I naively assumed that it was NP-Complete as the source code for Trade Dangerous [remarks](https://github.com/eyeonus/Trade-Dangerous/blob/7748478e4a19cfff9973b9c6c23b159167e2c1d6/tradedangerous/tradecalc.py#L657) that the algorithm used is *knapsack-like*. However, after discussing possible solutions to the trading route problem with professor [Abrahim Ladha](https://ladha.me/) (theoretical math/CS lecturer at Georgia Tech), it became evident that this problem could be solved with BFS.

More specifically, a graph of all systems are created where vertices represent systems and edges represent the fact that you can jump between the vertices (systems) of the edge. For each hop, a list of source vertices are provided (the first jump only has one). For each source vertex, all possible destination vertices in the hop are found through breadth first search (whose depth is constrained by a user-defined jumps/hop ratio). Then, for every station in the destination vertices, the maximum profit obtainable by hopping from a station in the source system/vertex to the destination station is recorded. This maximum solution is also max'd with the current global maximum solution for the corresponding destination system. Thus, after all traversals are complete, we are left with a list of stations and the maximum possible profit obtainable with that station as the destination. Furthermore, all visited systems (including this hop and previous hops) are passed to the next hop as source stations.

### Data
Market data is present in the form of a sqlite3 database (`TradeDangerous.db`) and can be downloaded via the Trade Dangerous [eddblink plugin](https://github.com/eyeonus/Trade-Dangerous/wiki/Plugin-Options#eddblink). For the sake of uniformity, the database used in testing and benchmarking is present on the repo.

### Parallelization
Computation of optimal solutions from a source system does not depend on the results of other traversals. Thus, parallelization is trivially achievable by splitting source systems among parallel executors (threads/processes). Node-level parallelization is achieved with OpenMP while internode parallelization is performed with MPI. Results must be combined into a global solution set, which is a serial operation. With OpenMP, results from multiple threads are combined *during* computation via locking solution entries and performing the max operation. On the other hand, the MPI solution adopts a combine step where solutions from non-root ranks are sent to the root rank which synchronously combines the process-level solutions into the global solution set.
