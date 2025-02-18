import json
import math
import random
import sys
from qubots.base_problem import BaseProblem
import os

PENALTY = -1e15

class PoolingProblem(BaseProblem):
    """
    Pooling Problem for Qubots.

    Raw materials (components) are blended to produce final products. Each component has a supply,
    a price, and quality attributes. Each product has a demand (lower bound), a capacity (upper bound),
    a sale price, and quality tolerance intervals. Intermediate pools allow blending before sending to products.
    
    Flows may occur directly from a component to a product (subject to an edge upper bound and cost) or indirectly:
    a fraction of a component's contribution is sent to a pool and then from the pool to the product.
    
    The objective is to maximize profit, defined as:
    
        Profit = (Total product revenue) - (Total cost of components) - (Edge costs)
    
    where
      - Total product revenue = sum_{p} (total inflow into product p * product price_p)
      - Total component cost = sum_{c} (total outflow from component c * component price_c)
      - Edge costs combine the direct cost on component→product edges and the sum of costs on component→pool and pool→product edges.
    
    A candidate solution is a dictionary with keys:
      - "component_to_product": 2D list of flows (dimensions: nbComponents × nbProducts)
      - "component_to_pool_fraction": 2D list of proportions (dimensions: nbComponents × nbPools)
      - "pool_to_product": 2D list of flows (dimensions: nbPools × nbProducts)
    """
    def __init__(self, instance_file: str, **kwargs):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(instance_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            instance_file = os.path.join(base_dir, instance_file)

        with open(instance_file, "r") as f:
            instance = json.load(f)
        
        # Basic counts
        self.nbComponents = len(instance["components"])
        self.nbProducts = len(instance["products"])
        self.nbAttributes = len(instance["components"][0]["quality"])
        self.nbPools = len(instance["pool_size"])
        
        # Components data
        self.componentPrices = [float(instance["components"][c]["price"]) for c in range(self.nbComponents)]
        self.componentSupplies = [float(instance["components"][c]["upper"]) for c in range(self.nbComponents)]
        self.componentQuality = [list(map(float, instance["components"][c]["quality"].values()))
                                  for c in range(self.nbComponents)]
        self.componentNames = [instance["components"][c]["name"] for c in range(self.nbComponents)]
        compIndex = {instance["components"][c]["name"]: c for c in range(self.nbComponents)}
        
        # Products data
        self.productPrices = [float(instance["products"][p]["price"]) for p in range(self.nbProducts)]
        self.productCapacities = [float(instance["products"][p]["upper"]) for p in range(self.nbProducts)]
        self.demand = [float(instance["products"][p]["lower"]) for p in range(self.nbProducts)]
        self.productNames = [instance["products"][p]["name"] for p in range(self.nbProducts)]
        prodIndex = {instance["products"][p]["name"]: p for p in range(self.nbProducts)}
        
        # Quality tolerance for products (if quality_lower is null, use 0)
        self.minTolerance = []
        for p in range(self.nbProducts):
            if instance["products"][p]["quality_lower"] is None:
                self.minTolerance.append([0.0]*self.nbAttributes)
            else:
                self.minTolerance.append(list(map(float, instance["products"][p]["quality_lower"].values())))
        self.maxTolerance = [list(map(float, instance["products"][p]["quality_upper"].values()))
                             for p in range(self.nbProducts)]
        
        # Pools data
        self.poolNames = list(instance["pool_size"].keys())
        self.poolCapacities = [float(instance["pool_size"][o]) for o in self.poolNames]
        poolIndex = {self.poolNames[o]: o for o in range(self.nbPools)}
        
        # Flow graph: initialize arrays with zeros.
        self.upperBoundComponentToProduct = [[0.0 for _ in range(self.nbProducts)] for _ in range(self.nbComponents)]
        self.costComponentToProduct = [[0.0 for _ in range(self.nbProducts)] for _ in range(self.nbComponents)]
        self.upperBoundFractionComponentToPool = [[0.0 for _ in range(self.nbPools)] for _ in range(self.nbComponents)]
        self.costComponentToPool = [[0.0 for _ in range(self.nbPools)] for _ in range(self.nbComponents)]
        self.upperBoundPoolToProduct = [[0.0 for _ in range(self.nbProducts)] for _ in range(self.nbPools)]
        self.costPoolToProduct = [[0.0 for _ in range(self.nbProducts)] for _ in range(self.nbPools)]
        
        # Fill component-to-product edges (if any)
        for edge in instance.get("component_to_product_bound", []):
            c = compIndex[edge["component"]]
            p = prodIndex[edge["product"]]
            self.upperBoundComponentToProduct[c][p] = float(edge["bound"])
            if "cost" in edge:
                self.costComponentToProduct[c][p] = float(edge["cost"])
        
        # Fill component-to-pool edges
        for edge in instance.get("component_to_pool_fraction", []):
            c = compIndex[edge["component"]]
            o = poolIndex[edge["pool"]]
            self.upperBoundFractionComponentToPool[c][o] = float(edge["fraction"])
            if "cost" in edge:
                self.costComponentToPool[c][o] = float(edge["cost"])
        
        # Fill pool-to-product edges
        for edge in instance.get("pool_to_product_bound", []):
            o = poolIndex[edge["pool"]]
            p = prodIndex[edge["product"]]
            self.upperBoundPoolToProduct[o][p] = float(edge["bound"])
            if "cost" in edge:
                self.costPoolToProduct[o][p] = float(edge["cost"])
        
    def evaluate_solution(self, solution) -> float:
        """
        Expects solution as a dictionary with keys:
          - "component_to_product": 2D list (nbComponents x nbProducts)
          - "component_to_pool_fraction": 2D list (nbComponents x nbPools)
          - "pool_to_product": 2D list (nbPools x nbProducts)
        
        Returns the profit if all constraints are satisfied; otherwise, returns a very low profit.
        """
        # Unpack candidate flows.
        try:
            cp = solution["component_to_product"]
            c2p = solution["component_to_pool_fraction"]
            pp = solution["pool_to_product"]
        except KeyError:
            return PENALTY

        # Check dimensions.
        if (len(cp) != self.nbComponents or any(len(row) != self.nbProducts for row in cp) or
            len(c2p) != self.nbComponents or any(len(row) != self.nbPools for row in c2p) or
            len(pp) != self.nbPools or any(len(row) != self.nbProducts for row in pp)):
            return PENALTY

        # Constraint 1: For each pool o, sum_{c} c2p[c][o] must equal 1.
        for o in range(self.nbPools):
            total_fraction = sum(c2p[c][o] for c in range(self.nbComponents))
            if abs(total_fraction - 1.0) > 1e-3:
                return PENALTY

        # Constraint 2: For each component c, total outflow <= componentSupplies.
        for c in range(self.nbComponents):
            direct = sum(cp[c][p] for p in range(self.nbProducts))
            indirect = 0.0
            for o in range(self.nbPools):
                for p in range(self.nbProducts):
                    indirect += c2p[c][o] * pp[o][p]
            if direct + indirect > self.componentSupplies[c] + 1e-6:
                return PENALTY

        # Constraint 3: For each component c and pool o,
        # sum_{p} (c2p[c][o] * pp[o][p]) <= poolCapacities[o] * c2p[c][o]
        for c in range(self.nbComponents):
            for o in range(self.nbPools):
                flow_via_pool = sum(c2p[c][o] * pp[o][p] for p in range(self.nbProducts))
                if flow_via_pool > self.poolCapacities[o] * c2p[c][o] + 1e-6:
                    return PENALTY

        # Constraint 4: For each product p, total inflow between direct and from pools is within [demand, capacity].
        for p in range(self.nbProducts):
            direct = sum(cp[c][p] for c in range(self.nbComponents))
            from_pools = sum(pp[o][p] for o in range(self.nbPools))
            total_in = direct + from_pools
            if total_in < self.demand[p] - 1e-6 or total_in > self.productCapacities[p] + 1e-6:
                return PENALTY

        # Constraint 5: For each product p and attribute k, quality must be within tolerance.
        for p in range(self.nbProducts):
            direct_attr = [0.0]*len(self.componentQuality[0])
            pool_attr = [0.0]*len(self.componentQuality[0])
            for c in range(self.nbComponents):
                for k in range(len(self.componentQuality[c])):
                    direct_attr[k] += self.componentQuality[c][k] * cp[c][p]
            for o in range(self.nbPools):
                for c in range(self.nbComponents):
                    for k in range(len(self.componentQuality[c])):
                        pool_attr[k] += self.componentQuality[c][k] * c2p[c][o] * pp[o][p]
            total_in = sum(cp[c][p] for c in range(self.nbComponents)) + sum(pp[o][p] for o in range(self.nbPools))
            for k in range(len(direct_attr)):
                # If total_in is zero, skip (should not happen due to constraint 4).
                if total_in < 1e-6:
                    continue
                if direct_attr[k] + pool_attr[k] < self.minTolerance[p][k] * total_in - 1e-6:
                    return PENALTY
                if direct_attr[k] + pool_attr[k] > self.maxTolerance[p][k] * total_in + 1e-6:
                    return PENALTY

        # Compute the objective profit.
        # Direct flow cost.
        directFlowCost = 0.0
        for c in range(self.nbComponents):
            for p in range(self.nbProducts):
                directFlowCost += self.costComponentToProduct[c][p] * cp[c][p]
        # Indirect flow cost.
        undirectFlowCost = 0.0
        for c in range(self.nbComponents):
            for o in range(self.nbPools):
                for p in range(self.nbProducts):
                    undirectFlowCost += (self.costComponentToPool[c][o] + self.costPoolToProduct[o][p]) * (c2p[c][o] * pp[o][p])
        # Total inflow per product.
        total_inflow = [0.0]*self.nbProducts
        for p in range(self.nbProducts):
            total_inflow[p] = sum(cp[c][p] for c in range(self.nbComponents)) + sum(pp[o][p] for o in range(self.nbPools))
        productsGain = sum(total_inflow[p] * self.productPrices[p] for p in range(self.nbProducts))
        # Total outflow per component.
        total_outflow = [0.0]*self.nbComponents
        for c in range(self.nbComponents):
            total_direct = sum(cp[c][p] for p in range(self.nbProducts))
            total_indirect = 0.0
            for o in range(self.nbPools):
                for p in range(self.nbProducts):
                    total_indirect += c2p[c][o] * pp[o][p]
            total_outflow[c] = total_direct + total_indirect
        componentsCost = sum(total_outflow[c] * self.componentPrices[c] for c in range(self.nbComponents))
        profit = productsGain - componentsCost - (directFlowCost + undirectFlowCost)
        return profit
    
    def random_solution(self):
        # Generate random flows for each decision variable within bounds.
        cp = [[random.uniform(0, self.upperBoundComponentToProduct[c][p])
               for p in range(self.nbProducts)]
              for c in range(self.nbComponents)]
        # For each pool o, generate random numbers for each component and then normalize to sum to 1.
        c2p = []
        for c in range(self.nbComponents):
            row = []
            for o in range(self.nbPools):
                row.append(random.uniform(0, self.upperBoundFractionComponentToPool[c][o]))
            c2p.append(row)
        # Normalize each column (for each pool).
        for o in range(self.nbPools):
            s = sum(c2p[c][o] for c in range(self.nbComponents))
            if s == 0:
                for c in range(self.nbComponents):
                    c2p[c][o] = 1.0 / self.nbComponents
            else:
                for c in range(self.nbComponents):
                    c2p[c][o] /= s
        pp = [[random.uniform(0, self.upperBoundPoolToProduct[o][p])
               for p in range(self.nbProducts)]
              for o in range(self.nbPools)]
        return {
            "component_to_product": cp,
            "component_to_pool_fraction": c2p,
            "pool_to_product": pp
        }

    # The following attributes were read from the instance:
    # - self.upperBoundComponentToProduct, self.costComponentToProduct
    # - self.upperBoundFractionComponentToPool, self.costComponentToPool
    # - self.upperBoundPoolToProduct, self.costPoolToProduct

    # If any of these arrays were not provided in the instance, we default them to 0.
    # (In practice, a complete instance should include them.)
    # For robustness, we can set missing bounds/costs to zero.
    # Here we assume that if an edge is missing in the instance, the corresponding bound and cost remain 0.

