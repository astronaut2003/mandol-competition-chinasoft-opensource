from __future__ import annotations
from enum import Enum
from collections import defaultdict
from ..representation import MemoryUnit

class RetrievalMethod(Enum):
    """Retrieval backends supported by Mandol."""
    BM25 = "bm25"
    COSINE_SIMILARITY = "cosine_similarity"
    SPLADE = "splade"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"
    GRAPH_CONTEXT_EXPANSION = "graph_context_expansion"


RETRIEVAL_METHOD_MAPPING = {
    "bm25": RetrievalMethod.BM25,
    "cosine": RetrievalMethod.COSINE_SIMILARITY,
    "cosine_similarity": RetrievalMethod.COSINE_SIMILARITY,
    "splade": RetrievalMethod.SPLADE,
    "graph": RetrievalMethod.GRAPH_TRAVERSAL,
    "graph_traversal": RetrievalMethod.GRAPH_TRAVERSAL,
    "graph_context_expansion": RetrievalMethod.GRAPH_CONTEXT_EXPANSION,
    "hybrid": RetrievalMethod.HYBRID,
}


class RetrievalResult:
    """Single retrieved memory unit with score and backend metadata."""
    
    def __init__(self, unit: MemoryUnit, score: float, method: RetrievalMethod, 
                 method_details: Optional[Dict[str, Any]] = None):
        self.unit = unit
        self.score = score
        self.method = method
        self.method_details = method_details or {}
        
    def __repr__(self):
        return f"RetrievalResult(uid={self.unit.uid}, score={self.score:.4f}, method={self.method.value})"


class MultiRetrievalResults:
    """Accumulator for results returned by multiple retrieval backends."""
    
    def __init__(self):
        self.results_by_method: Dict[RetrievalMethod, List[RetrievalResult]] = defaultdict(list)
        self.results_by_uid: Dict[str, Dict[RetrievalMethod, RetrievalResult]] = defaultdict(dict)
        self.all_units: Dict[str, MemoryUnit] = {}
        
    def add_result(self, result: RetrievalResult):
        """Add result."""
        self.results_by_method[result.method].append(result)
        self.results_by_uid[result.unit.uid][result.method] = result
        self.all_units[result.unit.uid] = result.unit
        
    def add_results(self, results: List[RetrievalResult]):
        """Add results."""
        for result in results:
            self.add_result(result)
    
    def get_methods_used(self) -> Set[RetrievalMethod]:
        """Return methods used."""
        return set(self.results_by_method.keys())
    
    def get_units_found_by_method(self, method: RetrievalMethod) -> List[MemoryUnit]:
        """Return units found by method."""
        return [result.unit for result in self.results_by_method[method]]
    
    def get_intersection_units(self) -> List[str]:
        """Return intersection units."""
        if not self.results_by_method:
            return []
        
        method_uids = [
            set(result.unit.uid for result in results) 
            for results in self.results_by_method.values()
        ]
        return list(set.intersection(*method_uids))
    
    def get_union_units(self) -> List[str]:
        """Return union units."""
        return list(self.all_units.keys())
    
    def get_score_for_unit(self, uid: str, method: RetrievalMethod) -> Optional[float]:
        """Return score for unit."""
        if uid in self.results_by_uid and method in self.results_by_uid[uid]:
            return self.results_by_uid[uid][method].score
        return None

