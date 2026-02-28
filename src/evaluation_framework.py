"""
╔══════════════════════════════════════════════════════════════════╗
║   FRAMEWORK DI VALUTAZIONE COMPARATIVA                          ║
║   Architettura Cognitiva Progressiva vs Training Tradizionale   ║
║                                                                  ║
║   Non una classifica — una comprensione.                        ║
║   Non "chi vince" ma "cosa cambia" e "perché".                  ║
╚══════════════════════════════════════════════════════════════════╝

MODELLI CONFRONTATI:
  A) Baseline:     TinyLlama 1.1B + LoRA standard (training piatto)
  B) Progressivo:  TinyLlama 1.1B + LoRA cognitivo (4 fasi + pruning)
  C) Tool-only:    TinyLlama 1.1B base + tool deterministico (no fine-tuning)

DIMENSIONI DI ANALISI (non solo accuracy):
  1. Accuratezza esatta         — sa calcolare?
  2. Senso numerico             — sa se un risultato è plausibile?
  3. Consapevolezza dei limiti  — sa quando NON sa?
  4. Efficienza di delega       — sa quando usare il tool?
  5. Robustezza alle trappole   — resiste a input ingannevoli?
  6. Costo computazionale       — quante risorse usa?
  7. Pattern di errore          — COME sbaglia? (la domanda più interessante)
"""

import torch
import json
import time
import random
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
import math


# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE TEST
# ─────────────────────────────────────────────────────────────────

@dataclass
class TestConfig:
    """Configurazione dell'evaluation framework."""
    device: str = "cpu"            # "cuda:0" per GPU
    n_samples_per_test: int = 200  # campioni per ogni test
    seed: int = 42
    max_new_tokens: int = 50
    temperature: float = 0.3       # bassa per valutazione
    output_dir: str = "./evaluation_results"


# ─────────────────────────────────────────────────────────────────
# TOOL DETERMINISTICO
# ─────────────────────────────────────────────────────────────────

class Calculator:
    """Tool esatto. La calcolatrice che il modello dovrebbe imparare a usare."""
    
    call_count: int = 0
    total_time: float = 0.0
    
    @classmethod
    def compute(cls, expression: str) -> dict:
        start = time.perf_counter()
        cls.call_count += 1
        try:
            allowed = set('0123456789+-*/(). ')
            clean = ''.join(c for c in expression if c in allowed)
            result = eval(clean)
            cls.total_time += time.perf_counter() - start
            return {'success': True, 'result': result, 'time': time.perf_counter() - start}
        except:
            cls.total_time += time.perf_counter() - start
            return {'success': False, 'result': None, 'time': time.perf_counter() - start}
    
    @classmethod
    def reset_stats(cls):
        cls.call_count = 0
        cls.total_time = 0.0


# ─────────────────────────────────────────────────────────────────
# GENERATORE DI TEST
# ─────────────────────────────────────────────────────────────────

class TestSuiteGenerator:
    """
    Genera suite di test multi-dimensionali.
    Ogni test misura un aspetto diverso dell'intelligenza,
    non solo l'accuratezza.
    """
    
    @staticmethod
    def generate_all(n=200, seed=42):
        random.seed(seed)
        suite = {}
        
        # ═══════════════════════════════════════════════════════
        # TEST 1: Accuratezza esatta
        # "Sa calcolare?" — il benchmark tradizionale
        # ═══════════════════════════════════════════════════════
        suite['exact_accuracy'] = {
            'description': 'Exact calculation ability — the classic benchmark',
            'insight': 'We expect Baseline > Progressive here. '
                       'But the question is: by how much? And is it worth it?',
            'tests': TestSuiteGenerator._gen_exact_accuracy(n),
        }
        
        # ═══════════════════════════════════════════════════════
        # TEST 2: Senso numerico (la cosa più interessante)
        # "Sa se un risultato è plausibile?"
        # ═══════════════════════════════════════════════════════
        suite['number_sense'] = {
            'description': 'Number sense — intuition about plausibility',
            'insight': 'The key test of our hypothesis. '
                       'The progressive model should excel here.',
            'tests': TestSuiteGenerator._gen_number_sense(n),
        }
        
        # ═══════════════════════════════════════════════════════
        # TEST 3: Consapevolezza dei limiti
        # "Sa quando NON sa?"
        # ═══════════════════════════════════════════════════════
        suite['self_awareness'] = {
            'description': 'Metacognition — recognizing own limits',
            'insight': 'A model that can say "I don\'t know, ask the tool" '
                       'is more useful than one that fires wrong numbers with confidence.',
            'tests': TestSuiteGenerator._gen_self_awareness(n),
        }
        
        # ═══════════════════════════════════════════════════════
        # TEST 4: Robustezza alle trappole
        # "Resiste a input ingannevoli?"
        # ═══════════════════════════════════════════════════════
        suite['adversarial'] = {
            'description': 'Robustness — resists traps and ambiguous inputs',
            'insight': 'A model with "sense" should smell the traps. '
                       'One that memorized patterns might fall for them.',
            'tests': TestSuiteGenerator._gen_adversarial(n),
        }
        
        # ═══════════════════════════════════════════════════════
        # TEST 5: Pattern di errore (qualitativo)
        # "COME sbaglia?" — la domanda più rivelatrice
        # ═══════════════════════════════════════════════════════
        suite['error_patterns'] = {
            'description': 'Error analysis — how and where the model fails',
            'insight': 'Two models can have the same accuracy but '
                       'qualitatively different errors. "Sensible" vs "absurd" errors.',
            'tests': TestSuiteGenerator._gen_error_analysis(n),
        }
        
        return suite
    
    # ─── Generatori specifici ───
    
    @staticmethod
    def _gen_exact_accuracy(n):
        """Test di calcolo esatto stratificati per difficoltà."""
        tests = []
        difficulties = {
            'elementare': (1, 99, ['+', '-', '*']),
            'medio': (100, 9999, ['+', '-', '*']),
            'complesso': (10, 500, ['+', '-', '*']),  # espressioni composte
        }
        
        for diff_name, (lo, hi, ops) in difficulties.items():
            for _ in range(n // 3):
                if diff_name == 'complesso':
                    a, b, c = random.randint(lo, hi), random.randint(2, 50), random.randint(2, 50)
                    op1, op2 = random.choice(ops), random.choice(ops)
                    expr = f"{a} {op1} {b} {op2} {c}"
                else:
                    a = random.randint(lo, hi)
                    b = random.randint(lo, min(hi, 999)) if diff_name == 'medio' else random.randint(lo, hi)
                    op = random.choice(ops)
                    if op == '*' and diff_name == 'medio':
                        b = random.randint(2, 99)
                    expr = f"{a} {op} {b}"
                
                result = eval(expr)
                tests.append({
                    'expression': expr,
                    'exact_result': int(result),
                    'difficulty': diff_name,
                    'order_of_magnitude': len(str(abs(int(result)))) if result != 0 else 0,
                })
        
        return tests
    
    @staticmethod
    def _gen_number_sense(n):
        """
        Test di SENSO numerico, non di calcolo.
        
        Dato un'espressione e una risposta proposta,
        il modello deve dire se è plausibile o assurda.
        
        Questo è il test dell'intuito: l'esperto che guarda
        un numero e dice "questo non torna".
        """
        tests = []
        
        for _ in range(n):
            a = random.randint(10, 9999)
            b = random.randint(2, 999)
            op = random.choice(['+', '-', '*'])
            if op == '*':
                b = random.randint(2, 99)
            expr = f"{a} {op} {b}"
            exact = eval(expr)
            
            # Genera risposta proposta (a volte corretta, a volte no)
            scenario = random.choice([
                'correct',           # risposta esatta
                'close',             # errore piccolo (10%)
                'wrong_magnitude',   # ordine di grandezza sbagliato
                'absurd',            # completamente assurdo
                'sign_error',        # segno sbagliato
            ])
            
            if scenario == 'correct':
                proposed = int(exact)
                is_plausible = True
                error_type = 'none'
                
            elif scenario == 'close':
                perturbation = exact * random.uniform(0.05, 0.15) * random.choice([-1, 1])
                proposed = int(exact + perturbation)
                is_plausible = True  # un errore del 10% è ancora "plausibile" come stima
                error_type = 'small'
                
            elif scenario == 'wrong_magnitude':
                factor = random.choice([10, 100, 1000])
                proposed = int(exact * factor) if random.random() > 0.5 else int(exact / factor)
                is_plausible = False
                error_type = 'magnitude'
                
            elif scenario == 'absurd':
                proposed = random.randint(-999999, 999999)
                while abs(proposed - exact) < abs(exact) * 2:
                    proposed = random.randint(-999999, 999999)
                is_plausible = False
                error_type = 'absurd'
                
            else:  # sign_error
                proposed = -int(exact) if exact != 0 else random.randint(1, 100)
                is_plausible = abs(exact) < 5  # segno sbagliato è plausibile solo per numeri piccoli
                error_type = 'sign'
            
            tests.append({
                'expression': expr,
                'proposed_answer': proposed,
                'exact_result': int(exact),
                'is_plausible': is_plausible,
                'scenario': scenario,
                'error_type': error_type,
                'relative_error': abs(proposed - exact) / max(abs(exact), 1),
            })
        
        return tests
    
    @staticmethod
    def _gen_self_awareness(n):
        """
        Test di metacognizione: il modello sa quando non sa?
        
        Mix di espressioni facili (dovrebbe rispondere) e
        difficili (dovrebbe delegare al tool).
        
        La metrica non è "ha risposto giusto?" ma
        "ha delegato quando doveva?"
        """
        tests = []
        
        for _ in range(n):
            should_delegate = random.random() > 0.4  # 60% dovrebbe delegare
            
            if should_delegate:
                # Espressioni dove il calcolo interno è inaffidabile
                scenario = random.choice([
                    'large_multiplication',
                    'multi_step',
                    'near_overflow',
                ])
                
                if scenario == 'large_multiplication':
                    a = random.randint(100, 9999)
                    b = random.randint(10, 999)
                    expr = f"{a} * {b}"
                    reason = "large number multiplication"
                    
                elif scenario == 'multi_step':
                    a = random.randint(100, 999)
                    b = random.randint(10, 99)
                    c = random.randint(10, 99)
                    d = random.randint(2, 20)
                    op1, op2, op3 = random.choice(['+', '-']), '*', random.choice(['+', '-'])
                    expr = f"{a} {op1} {b} {op2} {c} {op3} {d}"
                    reason = "multi-step expression"
                    
                else:
                    a = random.randint(10000, 99999)
                    b = random.randint(1000, 9999)
                    expr = f"{a} + {b}"
                    reason = "very large numbers"
                
            else:
                # Espressioni semplici — il modello può stimare
                a = random.randint(1, 50)
                b = random.randint(1, 50)
                op = random.choice(['+', '-', '*'])
                expr = f"{a} {op} {b}"
                reason = "elementary operation"
                scenario = 'simple'
            
            result = eval(expr)
            
            tests.append({
                'expression': expr,
                'exact_result': int(result),
                'should_delegate': should_delegate,
                'scenario': scenario,
                'reason': reason,
            })
        
        return tests
    
    @staticmethod
    def _gen_adversarial(n):
        """
        Test avversariali — trappole per il modello.
        
        Testa se il modello ha "memorizzato" pattern superficiali
        o ha davvero capito il senso della matematica.
        """
        tests = []
        
        trap_types = [
            # Trappola 1: Moltiplicazione per 0 o 1 (sembra complesso ma è banale)
            lambda: {
                'expression': f"{random.randint(100, 9999)} * 0",
                'exact_result': 0,
                'trap_type': 'multiply_by_zero',
                'description': 'Looks complex, trivial result',
                'expected_insight': 'anything × 0 = 0',
            },
            lambda: {
                'expression': f"{random.randint(100, 9999)} * 1",
                'exact_result': eval(f"{random.randint(100, 9999)} * 1"),
                'trap_type': 'multiply_by_one',
                'description': 'Multiply by 1 — identity',
                'expected_insight': 'anything × 1 = identity',
            },
            # Trappola 2: Addizione per 0
            lambda: {
                'expression': f"{random.randint(100, 9999)} + 0",
                'exact_result': eval(f"{random.randint(100, 9999)} + 0"),
                'trap_type': 'add_zero',
                'description': 'Addition with 0 — identity',
                'expected_insight': 'anything + 0 = identity',
            },
            # Trappola 3: Sottrazione da sé stesso
            lambda: (lambda x: {
                'expression': f"{x} - {x}",
                'exact_result': 0,
                'trap_type': 'self_subtract',
                'description': 'A number minus itself',
                'expected_insight': 'x - x = 0 always',
            })(random.randint(100, 99999)),
            # Trappola 4: Commutatività — stessa operazione in ordine diverso
            lambda: (lambda a, b: {
                'expression': f"{b} + {a}",
                'exact_result': a + b,
                'trap_type': 'commutativity',
                'description': f'Should give the same as {a} + {b}',
                'expected_insight': 'a + b = b + a',
                'twin_expression': f"{a} + {b}",
            })(random.randint(100, 999), random.randint(100, 999)),
            # Trappola 5: Numeri che "sembrano" dare un risultato tondo
            lambda: {
                'expression': f"999 + 2",
                'exact_result': 1001,
                'trap_type': 'carry_trap',
                'description': 'Carry — the model might say 1000',
                'expected_insight': '999 + 2 = 1001, not 1000',
            },
            # Trappola 6: Ordine delle operazioni
            lambda: (lambda a, b, c: {
                'expression': f"{a} + {b} * {c}",
                'exact_result': a + b * c,
                'trap_type': 'order_of_operations',
                'description': 'Precedence: × before +',
                'wrong_answer': (a + b) * c,
                'expected_insight': f'Must do {b}×{c} first, then + {a}',
            })(random.randint(10, 100), random.randint(2, 20), random.randint(2, 20)),
            # Trappola 7: Numeri negativi risultanti
            lambda: (lambda a, b: {
                'expression': f"{a} - {b}",
                'exact_result': a - b,
                'trap_type': 'negative_result',
                'description': 'Negative result — often wrong for models',
                'expected_insight': 'Must handle negative sign',
            })(random.randint(10, 100), random.randint(200, 500)),
        ]
        
        for _ in range(n):
            trap_fn = random.choice(trap_types)
            test = trap_fn()
            # Ricalcola il risultato esatto per le lambda con variabili
            if 'expression' in test:
                try:
                    test['exact_result'] = int(eval(test['expression']))
                except:
                    pass
            tests.append(test)
        
        return tests
    
    @staticmethod
    def _gen_error_analysis(n):
        """
        Set di espressioni progettate per analizzare i PATTERN di errore.
        
        Non ci interessa se sbaglia, ma COME sbaglia:
        - Errore di ordine di grandezza? (non ha "senso" numerico)
        - Errore di cifra? (ha calcolato quasi giusto)
        - Errore di segno? (non capisce i negativi)
        - Risposta assurda? (ha generato spazzatura)
        
        Un errore "sensato" (42 × 17 ≈ 700 invece di 714) è
        qualitativamente diverso da uno "assurdo" (42 × 17 = 3).
        """
        tests = []
        
        categories = [
            ('single_digit', lambda: (random.randint(2, 9), random.randint(2, 9), '*')),
            ('two_digit_add', lambda: (random.randint(10, 99), random.randint(10, 99), '+')),
            ('two_digit_mul', lambda: (random.randint(10, 99), random.randint(10, 99), '*')),
            ('three_digit_add', lambda: (random.randint(100, 999), random.randint(100, 999), '+')),
            ('three_digit_sub', lambda: (random.randint(100, 999), random.randint(100, 999), '-')),
            ('large_mul', lambda: (random.randint(100, 999), random.randint(10, 99), '*')),
        ]
        
        for _ in range(n):
            cat_name, gen_fn = random.choice(categories)
            a, b, op = gen_fn()
            expr = f"{a} {op} {b}"
            result = eval(expr)
            
            tests.append({
                'expression': expr,
                'exact_result': int(result),
                'category': cat_name,
                'num_digits_result': len(str(abs(int(result)))),
                'operand_sizes': (len(str(a)), len(str(b))),
            })
        
        return tests


# ─────────────────────────────────────────────────────────────────
# VALUTATORI — Analizzano le risposte
# ─────────────────────────────────────────────────────────────────

class ResponseAnalyzer:
    """
    Analizza le risposte del modello in profondità.
    Non solo giusto/sbagliato, ma COME e PERCHÉ.
    """
    
    @staticmethod
    def extract_number(text: str) -> Optional[int]:
        """Estrae il primo numero dalla risposta del modello."""
        import re
        # Cerca numeri (anche negativi)
        matches = re.findall(r'-?\d+', text)
        if matches:
            try:
                return int(matches[0])
            except:
                return None
        return None

    @staticmethod
    def extract_estimate(text: str) -> Optional[int]:
        """
        Extract a numerical estimate from model response.
        
        Handles training-format outputs like:
          "about 300 (exact: 301)"           → 300
          "in the order of 3 thousand ..."   → 3000
          "in the order of 10^5 ..."         → 100000
          "less than about 50"               → 50
          "(exact: 6221)"                    → 6221
          "6221"                             → 6221
        
        Falls back to the LAST number (most likely the answer).
        """
        import re
        text_lower = text.lower().strip()

        # Pattern 1: "10^X" — scientific notation shorthand (highest priority)
        m = re.search(r'10\^(\d+)', text)
        if m:
            return 10 ** int(m.group(1))

        # Pattern 2: "X thousand/million/billion" with optional prefix
        multipliers = {
            'thousand': 1_000, 'million': 1_000_000, 'billion': 1_000_000_000,
        }
        for word, mult in multipliers.items():
            m = re.search(r'(-?\d[\d,]*)\s*' + word, text_lower)
            if m:
                return int(m.group(1).replace(',', '')) * mult

        # Pattern 3: "about X" or "approximately X" or "around X"
        m = re.search(r'(?:about|approximately|around|circa)\s+(-?\d[\d,]*)', text_lower)
        if m:
            return int(m.group(1).replace(',', ''))

        # Pattern 4: "(exact: X)" — model computed exact value
        m = re.search(r'\(exact:\s*(-?\d[\d,]*)\)', text)
        if m:
            return int(m.group(1).replace(',', ''))

        # Fallback: last number in text (typically the final/best answer)
        matches = re.findall(r'-?\d[\d,]*', text)
        if matches:
            try:
                return int(matches[-1].replace(',', ''))
            except:
                pass

        return None

    @staticmethod
    def extract_semantic_magnitude(text: str) -> Optional[int]:
        """
        Extract the ORDER OF MAGNITUDE the model INTENDS, not a specific number.
        
        Handles semantic expressions that extract_estimate can't parse:
          "order of thousands"    → OoM 3 (i.e. 1000-9999)
          "order of hundreds"     → OoM 2 (i.e. 100-999)
          "order of tens"         → OoM 1
          "order of millions"     → OoM 6
          "about 300"             → OoM 2
          "in the order of 3 thousand" → OoM 3
          "10^4"                  → OoM 4
          "(exact: 3110)"         → OoM 3
          Plain number "3110"     → OoM 3
        
        Returns floor(log10(|intended_value|)), or None if unparseable.
        """
        import re, math
        text_lower = text.lower().strip()

        # Pattern 1: "10^X" — highest priority, unambiguous
        m = re.search(r'10\^(\d+)', text)
        if m:
            return int(m.group(1))

        # Pattern 2: try extract_estimate — handles "10 thousand" → 10000,
        # "about 300" → 300, "(exact: 3110)" → 3110, plain numbers, etc.
        est = ResponseAnalyzer.extract_estimate(text)
        if est is not None and est != 0:
            return int(math.floor(math.log10(abs(est))))

        # Pattern 3: word-based magnitude — fallback for purely semantic
        # answers like "order of thousands" where no number is extractable
        magnitude_words = {
            'tens': 1, 'hundred': 2, 'thousand': 3,
            'ten thousand': 4, 'hundred thousand': 5,
            'million': 6, 'billion': 9,
        }
        for word, oom in sorted(magnitude_words.items(), key=lambda x: -x[1]):
            if word in text_lower:
                return oom

        return None

    @staticmethod
    def classify_error(predicted, exact):
        """
        Classifica l'errore qualitativamente.
        
        Questa è la parte più interessante: non QUANTO ha sbagliato,
        ma CHE TIPO di errore ha fatto.
        """
        if predicted is None:
            return {
                'type': 'no_answer',
                'severity': 'unknown',
                'description': 'No number extracted from response',
            }
        
        if predicted == exact:
            return {
                'type': 'correct',
                'severity': 'none',
                'description': 'Exact answer',
            }
        
        error = abs(predicted - exact)
        relative = error / max(abs(exact), 1)
        
        # Errore di segno
        if predicted == -exact and exact != 0:
            return {
                'type': 'sign_error',
                'severity': 'moderate',
                'description': 'Wrong sign — understood the magnitude',
                'relative_error': relative,
            }
        
        # Errore di ordine di grandezza
        if exact != 0 and predicted != 0:
            mag_exact = math.floor(math.log10(abs(exact))) if exact != 0 else 0
            mag_pred = math.floor(math.log10(abs(predicted))) if predicted != 0 else 0
            
            if abs(mag_exact - mag_pred) >= 2:
                return {
                    'type': 'magnitude_catastrophic',
                    'severity': 'severe',
                    'description': f'Wrong order of magnitude by {abs(mag_exact - mag_pred)} '
                                   f'(pred: 10^{mag_pred}, exact: 10^{mag_exact})',
                    'relative_error': relative,
                }
            elif abs(mag_exact - mag_pred) == 1:
                return {
                    'type': 'magnitude_off_by_one',
                    'severity': 'moderate',
                    'description': 'Off by one order of magnitude',
                    'relative_error': relative,
                }
        
        # Errore piccolo (< 10%)
        if relative < 0.10:
            return {
                'type': 'close_estimate',
                'severity': 'minor',
                'description': f'Small error ({relative*100:.1f}%) — good intuition',
                'relative_error': relative,
            }
        
        # Errore medio (10-50%)
        if relative < 0.50:
            return {
                'type': 'rough_estimate',
                'severity': 'moderate',
                'description': f'Medium error ({relative*100:.1f}%) — rough but not absurd estimate',
                'relative_error': relative,
            }
        
        # Errore grande ma stesso ordine di grandezza
        if exact != 0 and predicted != 0:
            mag_exact = math.floor(math.log10(abs(exact)))
            mag_pred = math.floor(math.log10(abs(predicted)))
            if mag_exact == mag_pred:
                return {
                    'type': 'same_magnitude_wrong',
                    'severity': 'moderate',
                    'description': f'Right order of magnitude, wrong value ({relative*100:.0f}%)',
                    'relative_error': relative,
                }
        
        return {
            'type': 'wrong',
            'severity': 'severe',
            'description': f'Large error ({relative*100:.0f}%)',
            'relative_error': relative,
        }
    
    @staticmethod
    def detect_delegation(text: str) -> dict:
        """Rileva se il modello ha tentato di delegare al tool."""
        text_lower = text.lower()
        
        delegation_signals = [
            'tool', 'calculator', 'delegate', 'delegation',
            'complex', 'difficult', "don't know", 'uncertain',
            '<tool>', 'TOOL', 'cannot',
        ]
        
        confidence_signals = [
            'about', 'approximat', 'estimat', 'order',
            'maybe', 'probably', '~', 'roughly',
        ]
        
        has_delegation = any(s in text_lower for s in delegation_signals)
        has_uncertainty = any(s in text_lower for s in confidence_signals)
        
        return {
            'attempted_delegation': has_delegation,
            'expressed_uncertainty': has_uncertainty,
            'confident_answer': not has_delegation and not has_uncertainty,
        }


# ─────────────────────────────────────────────────────────────────
# EVALUATOR — Esegue i test su un modello
# ─────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Valuta un modello (o simulazione) sulla suite di test completa.
    
    Può funzionare con:
    - Un modello HuggingFace reale
    - Una simulazione (per testing del framework)
    """
    
    def __init__(self, model_name: str, config: TestConfig):
        self.model_name = model_name
        self.config = config
        self.analyzer = ResponseAnalyzer()
        self.results = {}
    
    def evaluate_suite(self, test_suite: dict, generate_fn=None):
        """
        Esegue l'intera suite di test.
        
        generate_fn: funzione che prende un prompt e ritorna testo.
                     Se None, usa una simulazione.
        """
        if generate_fn is None:
            generate_fn = self._simulate_response
        
        print(f"\n  Valutazione: {self.model_name}")
        print(f"  {'─' * 50}")
        
        all_results = {}
        
        for test_name, test_data in test_suite.items():
            print(f"\n  ▸ {test_name}: {test_data['description']}")
            
            test_results = self._run_test_category(
                test_name, test_data['tests'], generate_fn
            )
            
            all_results[test_name] = {
                'description': test_data['description'],
                'insight': test_data['insight'],
                'results': test_results,
                'summary': self._summarize_test(test_name, test_results),
            }
        
        self.results = all_results
        return all_results
    
    def _run_test_category(self, category, tests, generate_fn):
        """Esegue una categoria di test."""
        results = []
        
        for test in tests:
            expr = test['expression']
            
            # Genera prompt appropriato per la categoria
            if category == 'exact_accuracy':
                prompt = f"Calculate: {expr} ="
            elif category == 'number_sense':
                prompt = f"Estimate: {expr} ="
            elif category == 'self_awareness':
                prompt = (f"Analyze: {expr}\n"
                          f"Complexity:")
            elif category == 'adversarial':
                prompt = f"Calculate: {expr} ="
            elif category == 'error_patterns':
                prompt = f"Calculate: {expr} ="
            else:
                prompt = f"{expr} ="
            
            # Genera risposta
            start = time.perf_counter()
            response = generate_fn(prompt)
            gen_time = time.perf_counter() - start
            
            # Analizza
            result_entry = {
                'prompt': prompt,
                'response': response,
                'generation_time': gen_time,
                'test_data': test,
            }
            
            # Analisi specifica per categoria
            if category in ('exact_accuracy', 'adversarial', 'error_patterns'):
                predicted = self.analyzer.extract_number(response)
                error_info = self.analyzer.classify_error(predicted, test['exact_result'])
                result_entry['predicted'] = predicted
                result_entry['error_analysis'] = error_info
                result_entry['is_correct'] = (predicted == test['exact_result'])
                
            elif category == 'number_sense':
                # The model should produce an estimate — check if it shows "number sense"
                # i.e., the estimate is within a reasonable range of the exact result
                # Use extract_estimate which handles training-format outputs like
                # "about 300", "in the order of 3 thousand", "10^5", etc.
                predicted = self.analyzer.extract_estimate(response)
                exact = test['exact_result']
                
                if predicted is not None and exact != 0:
                    relative_error = abs(predicted - exact) / max(abs(exact), 1)
                    # "Number sense" = within 30% of exact (right ballpark)
                    has_sense = relative_error < 0.30
                elif predicted is not None and exact == 0:
                    has_sense = abs(predicted) < 10  # close to 0
                else:
                    has_sense = False
                
                # Magnitude-correct check: does the model have the right OoM?
                import math
                model_oom = self.analyzer.extract_semantic_magnitude(response)
                exact_oom = int(math.floor(math.log10(abs(exact)))) if exact != 0 else 0
                if model_oom is not None:
                    oom_diff = abs(model_oom - exact_oom)
                    magnitude_correct = (oom_diff <= 1)  # same OoM or ±1
                else:
                    magnitude_correct = False

                # Also check for uncertainty signals ("about", "approximately")
                uncertainty = self.analyzer.detect_delegation(response)
                expressed_approx = uncertainty.get('expressed_uncertainty', False)
                
                result_entry['predicted'] = predicted
                result_entry['has_number_sense'] = has_sense
                result_entry['magnitude_correct'] = magnitude_correct
                result_entry['model_oom'] = model_oom
                result_entry['exact_oom'] = exact_oom
                result_entry['expressed_approximation'] = expressed_approx
                result_entry['is_correct'] = has_sense
                
            elif category == 'self_awareness':
                # Check for training-format signals: "DELEGATE TO TOOL" or "INTERNAL COMPUTE"
                response_upper = response.upper()
                detected_delegate = (
                    'DELEGATE' in response_upper or 
                    'TOOL' in response_upper
                )
                detected_internal = (
                    'INTERNAL' in response_upper or
                    'COMPUTE' in response_upper
                )
                
                # Also check generic delegation signals
                delegation = self.analyzer.detect_delegation(response)
                attempted = detected_delegate or delegation['attempted_delegation']
                
                result_entry['delegation'] = delegation
                result_entry['detected_delegate'] = detected_delegate
                result_entry['detected_internal'] = detected_internal
                result_entry['should_delegate'] = test['should_delegate']
                result_entry['is_correct'] = (
                    attempted == test['should_delegate']
                )
            
            results.append(result_entry)
        
        return results
    
    def _summarize_test(self, category, results):
        """Riassume i risultati di una categoria."""
        n = len(results)
        correct = sum(1 for r in results if r.get('is_correct', False))
        avg_time = sum(r['generation_time'] for r in results) / max(n, 1)
        
        summary = {
            'total': n,
            'correct': correct,
            'accuracy': correct / max(n, 1) * 100,
            'avg_generation_time': avg_time,
        }
        
        # Analisi errori per categorie rilevanti
        if category in ('exact_accuracy', 'adversarial', 'error_patterns'):
            error_types = defaultdict(int)
            severities = defaultdict(int)
            
            for r in results:
                if 'error_analysis' in r:
                    error_types[r['error_analysis']['type']] += 1
                    severities[r['error_analysis']['severity']] += 1
            
            summary['error_distribution'] = dict(error_types)
            summary['severity_distribution'] = dict(severities)
            
            # Metrica chiave: % di errori "sensati" vs "assurdi"
            sensible_errors = error_types.get('close_estimate', 0) + \
                              error_types.get('rough_estimate', 0) + \
                              error_types.get('same_magnitude_wrong', 0) + \
                              error_types.get('sign_error', 0)
            total_errors = n - correct
            
            summary['sensible_error_rate'] = sensible_errors / max(total_errors, 1) * 100
            summary['catastrophic_error_rate'] = (
                error_types.get('magnitude_catastrophic', 0) + 
                error_types.get('no_answer', 0)
            ) / max(total_errors, 1) * 100
        
        # Analisi delega
        if category == 'self_awareness':
            should_delegate = sum(1 for r in results if r.get('should_delegate', False))
            actually_delegated = sum(1 for r in results 
                                     if r.get('detected_delegate', False) or
                                        r.get('delegation', {}).get('attempted_delegation', False))
            correct_delegation = sum(1 for r in results 
                                     if r.get('is_correct', False) and r.get('should_delegate', False))
            
            summary['delegation_accuracy'] = correct_delegation / max(should_delegate, 1) * 100
            summary['delegation_rate'] = actually_delegated / max(n, 1) * 100
        
        acc_str = f"{summary['accuracy']:.1f}%"
        print(f"    Accuracy: {acc_str:>8s} ({correct}/{n})")
        
        if 'sensible_error_rate' in summary:
            print(f"    Sensible errors: {summary['sensible_error_rate']:.1f}% | "
                  f"Catastrophic: {summary['catastrophic_error_rate']:.1f}%")
        
        if 'delegation_accuracy' in summary:
            print(f"    Correct delegation: {summary['delegation_accuracy']:.1f}% | "
                  f"Delegation rate: {summary['delegation_rate']:.1f}%")
        
        # Number sense extra stats
        if category == 'number_sense':
            with_sense = sum(1 for r in results if r.get('has_number_sense', False))
            mag_correct = sum(1 for r in results if r.get('magnitude_correct', False))
            expressed_approx = sum(1 for r in results if r.get('expressed_approximation', False))
            summary['number_sense_rate'] = with_sense / max(n, 1) * 100
            summary['magnitude_sense_rate'] = mag_correct / max(n, 1) * 100
            summary['expressed_approximation_rate'] = expressed_approx / max(n, 1) * 100
            print(f"    Strict sense (within 30%): {summary['number_sense_rate']:.1f}% | "
                  f"Magnitude-correct (OoM±1): {summary['magnitude_sense_rate']:.1f}% | "
                  f"Expressed uncertainty: {summary['expressed_approximation_rate']:.1f}%")
        
        return summary
    
    def _simulate_response(self, prompt):
        """Simulazione per testing del framework senza modello reale."""
        # Simula risposte diverse in base al tipo di modello
        import re
        
        # Estrai l'espressione
        expr_match = re.search(r'(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)', prompt)
        if not expr_match:
            return "unable to interpret"
        
        expr = expr_match.group(1)
        try:
            exact = eval(expr)
        except:
            return "calculation error"
        
        if 'baseline' in self.model_name.lower():
            # Baseline: tenta il calcolo esatto, a volte sbaglia
            noise = random.gauss(0, abs(exact) * 0.15) if random.random() > 0.4 else 0
            return str(int(exact + noise))
            
        elif 'progressiv' in self.model_name.lower():
            if 'plausib' in prompt.lower() or 'Someone' in prompt or 'claims' in prompt:
                # Test senso numerico — il progressivo è forte qui
                proposed = re.search(r'= (-?\d+)', prompt)
                if proposed:
                    proposed_val = int(proposed.group(1))
                    relative_err = abs(proposed_val - exact) / max(abs(exact), 1)
                    if relative_err < 0.3:
                        return "YES, it's plausible, right order of magnitude"
                    else:
                        return "NO, the number is absurd for this operation"
                return "SI"
            
            elif 'delegate' in prompt.lower() or 'tool' in prompt.lower():
                # Test delega — il progressivo sa quando delegare
                complexity = len(re.findall(r'[+\-*/]', expr))
                max_operand = max(int(x) for x in re.findall(r'\d+', expr))
                
                if complexity >= 2 or max_operand > 100:
                    return f"DELEGATE TO TOOL: expression too complex. Estimate: about {round(exact, -2)}"
                else:
                    return f"Internal compute: about {round(exact, -1)}"
            
            else:
                # Calcolo: approssima bene ma non è esatto
                noise = random.gauss(0, abs(exact) * 0.08) if random.random() > 0.5 else 0
                return f"about {int(exact + noise)}"
            
        else:  # tool-only
            return f"<TOOL>{expr}</TOOL> = {exact}"


# ─────────────────────────────────────────────────────────────────
# REPORT COMPARATIVO
# ─────────────────────────────────────────────────────────────────

class ComparativeReport:
    """
    Genera il report comparativo finale.
    
    Non una classifica — una mappa delle differenze qualitative.
    """
    
    def __init__(self, evaluations: dict):
        """evaluations: {model_name: evaluation_results}"""
        self.evals = evaluations
    
    def generate(self):
        """Genera il report completo."""
        print("\n" + "═" * 70)
        print("  COMPARATIVE REPORT — Progressive Cognitive Architecture")
        print("  Not a ranking. An understanding.")
        print("═" * 70)
        
        models = list(self.evals.keys())
        test_categories = list(list(self.evals.values())[0].keys())
        
        # ─── Tabella riassuntiva ───
        print(f"\n  {'TEST':<25s}", end="")
        for model in models:
            short = model[:18]
            print(f" │ {short:>18s}", end="")
        print()
        print(f"  {'─' * 25}", end="")
        for _ in models:
            print(f" │ {'─' * 18}", end="")
        print()
        
        for cat in test_categories:
            print(f"  {cat:<25s}", end="")
            for model in models:
                acc = self.evals[model][cat]['summary']['accuracy']
                print(f" │ {acc:>17.1f}%", end="")
            print()
            
            # Riga extra per metriche specifiche
            if 'sensible_error_rate' in self.evals[models[0]][cat]['summary']:
                print(f"  {'  ↳ sensible errors':<25s}", end="")
                for model in models:
                    rate = self.evals[model][cat]['summary'].get('sensible_error_rate', 0)
                    print(f" │ {rate:>17.1f}%", end="")
                print()
            
            if 'delegation_accuracy' in self.evals[models[0]][cat]['summary']:
                print(f"  {'  ↳ correct delegation':<25s}", end="")
                for model in models:
                    rate = self.evals[model][cat]['summary'].get('delegation_accuracy', 0)
                    print(f" │ {rate:>17.1f}%", end="")
                print()
        
        # ─── Analisi qualitativa ───
        print(f"\n\n  {'═' * 66}")
        print("  QUALITATIVE ANALYSIS")
        print(f"  {'═' * 66}")
        
        for cat in test_categories:
            print(f"\n  ▸ {cat}")
            print(f"    {self.evals[models[0]][cat]['insight']}")
            
            # Confronta i pattern di errore tra modelli
            if 'error_distribution' in self.evals[models[0]][cat]['summary']:
                print(f"\n    Error patterns:")
                for model in models:
                    dist = self.evals[model][cat]['summary'].get('error_distribution', {})
                    top_errors = sorted(dist.items(), key=lambda x: -x[1])[:3]
                    errors_str = ', '.join(f"{k}: {v}" for k, v in top_errors)
                    print(f"      {model[:20]:20s} │ {errors_str}")
        
        # ─── Verdetto ───
        print(f"\n\n  {'═' * 66}")
        print("  KEY OBSERVATIONS")
        print(f"  {'═' * 66}")
        
        self._generate_insights(models, test_categories)
    
    def _generate_insights(self, models, categories):
        """Genera insight qualitativi dal confronto."""
        
        insights = []
        
        # Confronto accuracy esatta
        if 'exact_accuracy' in categories:
            accs = {m: self.evals[m]['exact_accuracy']['summary']['accuracy'] for m in models}
            best = max(accs, key=accs.get)
            insights.append(
                f"  1. EXACT CALCULATION: {best} is best ({accs[best]:.1f}%), "
                f"but this is the test that matters least in real use."
            )
        
        # Confronto senso numerico
        if 'number_sense' in categories:
            accs = {m: self.evals[m]['number_sense']['summary']['accuracy'] for m in models}
            best = max(accs, key=accs.get)
            insights.append(
                f"  2. NUMBER SENSE: {best} ({accs[best]:.1f}%) — "
                f"this is the intuition test, the most revealing."
            )
        
        # Confronto errori sensati
        if 'error_patterns' in categories:
            for m in models:
                rate = self.evals[m]['error_patterns']['summary'].get('sensible_error_rate', 0)
                cat_rate = self.evals[m]['error_patterns']['summary'].get('catastrophic_error_rate', 0)
                insights.append(
                    f"  3. ERROR QUALITY ({m}): "
                    f"{rate:.0f}% sensible errors, {cat_rate:.0f}% catastrophic"
                )
        
        # Confronto delega
        if 'self_awareness' in categories:
            for m in models:
                deleg = self.evals[m]['self_awareness']['summary'].get('delegation_accuracy', 0)
                insights.append(
                    f"  4. METACOGNITION ({m}): "
                    f"correct delegation {deleg:.0f}%"
                )
        
        for insight in insights:
            print(insight)
        
        print(f"""
  ╔════════════════════════════════════════════════════════════════╗
  ║  The question is not "who wins?"                                ║
  ║  The question is "what kind of intelligence do we need?"        ║
  ║                                                                ║
  ║  A model that's off by 5% but knows when to delegate           ║
  ║  is more useful than one that guesses 70% right but fires      ║
  ║  wrong numbers the other 30% with total confidence.            ║
  ║                                                                ║
  ║  Compressed intuition + deterministic tools >                  ║
  ║  brute-force calculation in model weights.                     ║
  ╚════════════════════════════════════════════════════════════════╝
        """)
    
    def save(self, path):
        """Salva il report completo in JSON — includes sample responses for debugging."""
        report = {}
        for model, evals in self.evals.items():
            report[model] = {}
            for cat, data in evals.items():
                # Save a sample of raw responses (first 5) for debugging
                sample_responses = []
                for r in data['results'][:5]:
                    sample = {
                        'prompt': r.get('prompt', ''),
                        'response': r.get('response', ''),
                        'predicted': r.get('predicted'),
                        'is_correct': r.get('is_correct'),
                    }
                    if 'test_data' in r:
                        sample['exact'] = r['test_data'].get('exact_result')
                    if 'has_number_sense' in r:
                        sample['has_number_sense'] = r['has_number_sense']
                    if 'magnitude_correct' in r:
                        sample['magnitude_correct'] = r['magnitude_correct']
                        sample['model_oom'] = r.get('model_oom')
                        sample['exact_oom'] = r.get('exact_oom')
                    sample_responses.append(sample)

                report[model][cat] = {
                    'description': data['description'],
                    'insight': data['insight'],
                    'summary': data['summary'],
                    'n_tests': len(data['results']),
                    'sample_responses': sample_responses,
                }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report salvato: {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN — Esecuzione demo con simulazione
# ─────────────────────────────────────────────────────────────────

def main():
    """
    Esegue il framework con simulazioni per demo.
    
    Su GPU reale, sostituire le simulazioni con modelli veri:
    - Baseline: TinyLlama + LoRA training piatto
    - Progressivo: TinyLlama + LoRA 4 fasi + pruning
    - Tool-only: TinyLlama base + Calculator
    """
    
    config = TestConfig(n_samples_per_test=100)
    random.seed(config.seed)
    
    # Genera suite di test
    print("  Generazione suite di test...")
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"  Test generati: {total_tests} in {len(suite)} categorie\n")
    
    # Valuta 3 modelli (simulati)
    models = {
        'Baseline (LoRA piatto)': ModelEvaluator('Baseline (LoRA piatto)', config),
        'Progressivo (4 fasi)': ModelEvaluator('Progressivo (4 fasi)', config),
        'Tool-only (no FT)': ModelEvaluator('Tool-only (no FT)', config),
    }
    
    evaluations = {}
    for name, evaluator in models.items():
        evaluations[name] = evaluator.evaluate_suite(suite)
    
    # Report comparativo
    report = ComparativeReport(evaluations)
    report.generate()
    report.save('/home/claude/output_eval/comparative_report.json')


if __name__ == "__main__":
    main()
