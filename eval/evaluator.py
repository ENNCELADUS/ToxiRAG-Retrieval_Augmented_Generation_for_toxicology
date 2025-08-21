"""
ToxiRAG Evaluation Logic
Implements grounding and citation coverage checks for toxicology responses.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from llm.agentic_pipeline import create_agentic_response
from retriever.retriever import search_documents
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single question."""
    question_id: str
    query: str
    response_text: str
    citations_found: List[str]
    expected_citations: List[Dict[str, Any]]
    grounding_score: float
    citation_coverage: float
    phrase_coverage: float
    overall_score: float
    passed: bool
    details: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Overall evaluation results."""
    dataset_name: str
    total_questions: int
    passed_questions: int
    average_grounding_score: float
    average_citation_coverage: float
    average_phrase_coverage: float
    overall_pass_rate: float
    question_results: List[EvaluationMetrics]
    config: Dict[str, Any]


class ToxiRAGEvaluator:
    """Main evaluator for ToxiRAG responses."""
    
    def __init__(self, config_path: Path):
        """Initialize evaluator with config."""
        self.config_path = config_path
        self.config = self._load_config()
        self.golden_questions = self._load_golden_questions()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_golden_questions(self) -> Dict[str, Any]:
        """Load golden questions dataset."""
        golden_path = Path(self.config['golden_questions_file'])
        with open(golden_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def extract_citations(self, response_text: str) -> List[str]:
        """Extract citations from response text in format [E1 · section_tag]."""
        pattern = r'\[E\d+ · [^\]]+\]'
        citations = re.findall(pattern, response_text)
        return citations
    
    def calculate_grounding_score(self, response_text: str, evidence_text: str) -> float:
        """Calculate how well the response is grounded in evidence."""
        if not evidence_text or not response_text:
            return 0.0
        
        # Simple overlap-based grounding score
        # In production, this could use RAGAS faithfulness score
        response_words = set(response_text.lower().split())
        evidence_words = set(evidence_text.lower().split())
        
        if len(response_words) == 0:
            return 0.0
        
        overlap = len(response_words.intersection(evidence_words))
        grounding_score = overlap / len(response_words)
        return min(1.0, grounding_score)
    
    def calculate_citation_coverage(self, 
                                  citations_found: List[str], 
                                  expected_citations: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Calculate citation coverage against expected citations."""
        if not expected_citations:
            return 1.0, {"reason": "No expected citations"}
        
        matches = []
        details = {"expected": len(expected_citations), "found": len(citations_found), "matches": []}
        
        for expected in expected_citations:
            expected_doc = expected.get('document_title', '')
            expected_section = expected.get('section_tag', '')
            
            for found_citation in citations_found:
                # Extract document and section from found citation
                # Format: [E1 · section_tag]
                if self._citation_matches_expected(found_citation, expected_doc, expected_section):
                    matches.append({
                        "found": found_citation,
                        "expected_doc": expected_doc,
                        "expected_section": expected_section
                    })
                    details["matches"].append(matches[-1])
                    break
        
        coverage = len(matches) / len(expected_citations) if expected_citations else 0.0
        details["coverage"] = coverage
        return coverage, details
    
    def _citation_matches_expected(self, found_citation: str, expected_doc: str, expected_section: str) -> bool:
        """Check if found citation matches expected citation."""
        # For now, match by section tag only since document title isn't in citation format
        # In production, could cross-reference with retrieval results
        if expected_section in found_citation:
            return True
        return False
    
    def calculate_phrase_coverage(self, 
                                response_text: str, 
                                expected_phrases: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Calculate coverage of required phrases in response."""
        if not expected_phrases:
            return 1.0, {"reason": "No required phrases"}
        
        response_lower = response_text.lower()
        found_phrases = []
        
        for phrase in expected_phrases:
            if phrase.lower() in response_lower:
                found_phrases.append(phrase)
        
        coverage = len(found_phrases) / len(expected_phrases) if expected_phrases else 0.0
        details = {
            "expected": expected_phrases,
            "found": found_phrases,
            "coverage": coverage
        }
        
        return coverage, details
    
    async def evaluate_question(self, 
                              question: Dict[str, Any], 
                              config: Dict[str, Any]) -> EvaluationMetrics:
        """Evaluate a single question."""
        question_id = question['id']
        query = question['question']
        expected_citations = question.get('expected_citations', [])
        min_citations_required = question.get('min_citations_required', 1)
        
        logger.info(f"Evaluating question {question_id}: {query[:50]}...")
        
        try:
            # Create agentic response
            agentic_config = {
                "openai_api_key": None,  # Will use from env
                "google_api_key": None,  # Will use from env
                "llm_provider": "openai",
                "temperature": 0.1,
                "top_k_docs": config.get('retrieval', {}).get('top_k', 5)
            }
            
            response = await create_agentic_response(
                query=query,
                config=agentic_config,
                collection_name=config.get('collection_name', 'toxicology_docs')
            )
            
            # Extract citations from response
            citations_found = self.extract_citations(response.response_text)
            
            # Calculate grounding score
            grounding_score = self.calculate_grounding_score(
                response.response_text, 
                response.evidence_pack.evidence_text
            )
            
            # Calculate citation coverage
            citation_coverage, citation_details = self.calculate_citation_coverage(
                citations_found, expected_citations
            )
            
            # Calculate phrase coverage
            all_required_phrases = []
            for exp_cite in expected_citations:
                all_required_phrases.extend(exp_cite.get('must_include_phrases', []))
            
            phrase_coverage, phrase_details = self.calculate_phrase_coverage(
                response.response_text, all_required_phrases
            )
            
            # Calculate overall score
            weights = {"grounding": 0.4, "citation": 0.3, "phrase": 0.3}
            overall_score = (
                weights["grounding"] * grounding_score +
                weights["citation"] * citation_coverage +
                weights["phrase"] * phrase_coverage
            )
            
            # Determine if passed
            min_grounding = config.get('scoring', {}).get('min_grounding_score', 0.5)
            min_citation = config.get('scoring', {}).get('min_citation_coverage', 0.5)
            min_overall = config.get('scoring', {}).get('min_overall_score', 0.6)
            
            passed = (
                grounding_score >= min_grounding and
                citation_coverage >= min_citation and
                overall_score >= min_overall and
                len(citations_found) >= min_citations_required
            )
            
            details = {
                "citation_details": citation_details,
                "phrase_details": phrase_details,
                "response_length": len(response.response_text),
                "evidence_length": len(response.evidence_pack.evidence_text),
                "reasoning_steps": len(response.reasoning_steps),
                "refusal_reason": response.refusal_reason
            }
            
            return EvaluationMetrics(
                question_id=question_id,
                query=query,
                response_text=response.response_text,
                citations_found=citations_found,
                expected_citations=expected_citations,
                grounding_score=grounding_score,
                citation_coverage=citation_coverage,
                phrase_coverage=phrase_coverage,
                overall_score=overall_score,
                passed=passed,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate question {question_id}: {str(e)}")
            return EvaluationMetrics(
                question_id=question_id,
                query=query,
                response_text=f"ERROR: {str(e)}",
                citations_found=[],
                expected_citations=expected_citations,
                grounding_score=0.0,
                citation_coverage=0.0,
                phrase_coverage=0.0,
                overall_score=0.0,
                passed=False,
                details={"error": str(e)}
            )
    
    async def run_evaluation(self, 
                           output_dir: Path,
                           llm_provider: str = "openai",
                           limit: Optional[int] = None) -> EvaluationResult:
        """Run full evaluation on golden questions."""
        logger.info(f"Starting evaluation on {len(self.golden_questions['questions'])} questions")
        
        # Update config with runtime parameters
        eval_config = self.config.copy()
        eval_config['llm_provider'] = llm_provider
        
        questions = self.golden_questions['questions']
        if limit:
            questions = questions[:limit]
        
        # Evaluate each question
        question_results = []
        for question in questions:
            result = await self.evaluate_question(question, eval_config)
            question_results.append(result)
            
            logger.info(f"Question {result.question_id}: "
                       f"Overall={result.overall_score:.3f}, "
                       f"Grounding={result.grounding_score:.3f}, "
                       f"Citations={result.citation_coverage:.3f}, "
                       f"Passed={result.passed}")
        
        # Calculate aggregate metrics
        total_questions = len(question_results)
        passed_questions = sum(1 for r in question_results if r.passed)
        
        avg_grounding = sum(r.grounding_score for r in question_results) / total_questions
        avg_citation = sum(r.citation_coverage for r in question_results) / total_questions
        avg_phrase = sum(r.phrase_coverage for r in question_results) / total_questions
        pass_rate = passed_questions / total_questions
        
        evaluation_result = EvaluationResult(
            dataset_name=self.golden_questions.get('dataset', 'unknown'),
            total_questions=total_questions,
            passed_questions=passed_questions,
            average_grounding_score=avg_grounding,
            average_citation_coverage=avg_citation,
            average_phrase_coverage=avg_phrase,
            overall_pass_rate=pass_rate,
            question_results=question_results,
            config=eval_config
        )
        
        # Save results
        self._save_results(evaluation_result, output_dir)
        
        logger.info(f"Evaluation completed: {passed_questions}/{total_questions} passed "
                   f"({pass_rate:.1%} pass rate)")
        
        return evaluation_result
    
    def _save_results(self, result: EvaluationResult, output_dir: Path):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "dataset": result.dataset_name,
            "total_questions": result.total_questions,
            "passed_questions": result.passed_questions,
            "pass_rate": result.overall_pass_rate,
            "average_scores": {
                "grounding": result.average_grounding_score,
                "citation_coverage": result.average_citation_coverage,
                "phrase_coverage": result.average_phrase_coverage
            },
            "config": result.config
        }
        
        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        detailed_results = []
        for qr in result.question_results:
            detailed_results.append({
                "question_id": qr.question_id,
                "query": qr.query,
                "response_text": qr.response_text,
                "citations_found": qr.citations_found,
                "expected_citations": qr.expected_citations,
                "scores": {
                    "grounding": qr.grounding_score,
                    "citation_coverage": qr.citation_coverage,
                    "phrase_coverage": qr.phrase_coverage,
                    "overall": qr.overall_score
                },
                "passed": qr.passed,
                "details": qr.details
            })
        
        with open(output_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_dir}")


async def run_evaluation(eval_config: Path,
                        collection_name: str,
                        llm_provider: str = "openai",
                        output_dir: Path = Path("eval/results"),
                        limit: Optional[int] = None) -> EvaluationResult:
    """Convenience function to run evaluation."""
    evaluator = ToxiRAGEvaluator(eval_config)
    
    # Override collection name if provided
    evaluator.config['collection_name'] = collection_name
    
    return await evaluator.run_evaluation(
        output_dir=output_dir,
        llm_provider=llm_provider,
        limit=limit
    )
