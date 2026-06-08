import pandas as pd
from typing import Dict, Any, Union
from rich.console import Console

from finalise.config import Config
from finalise.target import TargetAnalysis
from finalise.predictors import PredictorAnalysis
from finalise.model import Model

class AnalysisResult:
    def __init__(self, target: TargetAnalysis, predictors: PredictorAnalysis, model: Model):
        self.target = target
        self.predictors = predictors
        self.model = model

def analyze(prices_dict: dict[str, pd.Series], config: Config) -> AnalysisResult:
    """
    Orquestrador principal. Executa a pipeline completa:
    1. Análise do Alvo (TargetAnalysis)
    2. Seleção de Preditores (PredictorAnalysis)
    3. Treinamento e Validação do Modelo (Model)
    """
    console = Console()
    
    console.print("[bold cyan]Iniciando Pipeline de Análise (finalise)[/bold cyan]")
    
    # 1. Target Analysis
    console.print("\n[bold yellow]Fase 1: Análise do Alvo[/bold yellow]")
    ta = TargetAnalysis(prices_dict, config)
    try:
        ta.run()
        if ta.horizon is None:
            # Fallback for warning only mode
            if not config.force_continue:
                raise ValueError("Nenhum sinal detectado no ativo alvo e force_continue é False.")
            else:
                console.print("[bold red]Aviso: Nenhum sinal forte detectado, continuando com a melhor escala disponível.[/bold red]")
                ta.horizon = list(ta.results.keys())[0] if ta.results else config.horizons[0]
                ta.alpha = config.alpha
    except Exception as e:
        console.print(f"[bold red]Erro na Fase 1: {e}[/bold red]")
        raise e
        
    console.print(f"Horizonte selecionado: {ta.horizon} dias")
    console.print(f"Nível de significância (alpha): {ta.alpha}")
    
    # 2. Predictor Analysis
    console.print("\n[bold yellow]Fase 2: Seleção de Preditores[/bold yellow]")
    # Get the target series at the selected horizon
    if ta.horizon not in ta.results or "returns" not in ta.results[ta.horizon]:
        from finalise.target.returns import compute
        target_series = compute(prices_dict[config.target_ticker], ta.horizon, overlapping=False)
    else:
        target_series = ta.results[ta.horizon]["returns"]
        
    pa = PredictorAnalysis(prices_dict, target_series, ta.horizon, config, alpha=ta.alpha)
    pa.run()
    
    console.print(f"Preditores selecionados: {len(pa.selected)}")
    
    if len(pa.selected) == 0:
        console.print("[bold red]Aviso: Nenhum preditor foi selecionado. O pipeline será encerrado.[/bold red]")
        return AnalysisResult(ta, pa, None)
        
    # 3. Model
    console.print("\n[bold yellow]Fase 3: Modelagem Preditiva e Validação[/bold yellow]")
    m = Model(pa.selected, target_series, config)
    m.run()
    
    status = "[green]VÁLIDO[/green]" if m.is_outlier else "[red]INVÁLIDO[/red]"
    console.print(f"Resultado do Modelo: {status} (z-score = {m.z_score:.2f})")
    
    console.print("\n[bold cyan]Pipeline Concluído com Sucesso[/bold cyan]")
    
    return AnalysisResult(ta, pa, m)
