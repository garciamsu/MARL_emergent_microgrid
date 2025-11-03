#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer_utils.py
==================

Utilidades comunes para los optimizadores de hiperparámetros de todos los agentes.
Incluye las reglas de "acción correcta", cálculo de márgenes, y funciones auxiliares.
"""

from typing import Dict, Tuple, Callable
import numpy as np


# ==================================================
# Reglas de Acción Correcta (Configurable)
# ==================================================

def get_correct_action_battery(state: Dict) -> int:
    """
    Define la acción "ideal" para el agente Batería.

    Args:
        state: Dict con 'delta_p_interno', 'soc_idx', 'soc_max'
    
    Returns:
        Acción correcta: 0 (inactivo), 1 (cargar), 2 (descargar)
    """
    delta_p_interno = state['delta_p_interno']
    soc_idx = state['soc_idx']
    soc_max = state.get('soc_max', 4)

    # 1. Caso de Déficit
    if delta_p_interno < 0:
        if soc_idx > 0:
            return 2  # Descargar (si tiene carga)
        else:
            return 0  # Inactivo (no puede hacer nada)

    # 2. Caso de Excedente
    elif delta_p_interno > 0:
        if soc_idx < soc_max:
            return 1  # Cargar (si tiene espacio)
        else:
            return 0  # Inactivo (está llena)

    # 3. Caso de Equilibrio Perfecto
    else:  # delta_p_interno == 0
        return 0  # Inactivo


def get_correct_action_grid(state: Dict) -> int:
    """
    Define la acción "ideal" para el agente Red (Utility Grid).

    Args:
        state: Dict con 'delta_p_interno', 'soc_idx'
    
    Returns:
        Acción correcta: 0 (no suministrar), 1 (suministrar)
    """
    delta_p_interno = state['delta_p_interno']
    soc_idx = state['soc_idx']

    # La red SÓLO debe actuar si la microred está en emergencia
    # (Déficit interno Y sin batería)
    if delta_p_interno <= 0 and soc_idx == 0:
        return 1  # Suministrar
    else:
        return 0  # No Suministrar


def get_correct_action_solar(state: Dict) -> int:
    """
    Define la acción "ideal" para el agente Solar (PV).

    Args:
        state: Dict con 'solar_potential_idx'
    
    Returns:
        Acción correcta: 0 (no suministrar), 1 (suministrar)
    """
    solar_potential_idx = state['solar_potential_idx']

    if solar_potential_idx > 0:
        return 1  # Suministrar
    else:
        return 0  # No Suministrar


def get_correct_action_wind(state: Dict) -> int:
    """
    Define la acción "ideal" para el agente Eólico (Wind).

    Args:
        state: Dict con 'wind_potential_idx'
    
    Returns:
        Acción correcta: 0 (no suministrar), 1 (suministrar)
    """
    wind_potential_idx = state['wind_potential_idx']

    if wind_potential_idx > 0:
        return 1  # Suministrar
    else:
        return 0  # No Suministrar


def get_correct_action_load(state: Dict) -> int:
    """
    Define la acción "ideal" para el Agente de Carga (Load Agent).

    Args:
        state: Dict con 'soc_idx', 'delta_p_interno', 'price', 'comfort_threshold'
    
    Returns:
        Acción correcta: 0 (apagar), 1 (encender)
    """
    soc_idx = state['soc_idx']
    delta_p_interno = state['delta_p_interno']
    price = state['price']
    comfort_threshold = state.get('comfort_threshold', 4)

    # Caso 1: Hay energía interna disponible
    # (Batería tiene carga O hay excedente renovable)
    if soc_idx > 0 or delta_p_interno > 0:
        return 1  # Encender (consumir energía gratis)

    # Caso 2: NO hay energía interna
    else:
        # Caso 2.A: El precio es barato
        if price <= comfort_threshold:
            return 1  # Encender (comprar barato)
        # Caso 2.B: El precio es caro
        else:
            return 0  # Apagar (evitar costo alto)


# ==================================================
# Cálculo de Márgenes
# ==================================================

def compute_margin_for_state(
    state_dict: Dict,
    all_actions: list,
    reward_fn: Callable,
    agent_mock,
    env_mock,
    state_tuple: tuple,
    get_correct_action_fn: Callable
) -> float:
    """
    Calcula el margen para un estado dado.
    
    Margen = recompensa(acción_correcta) - max(recompensa(acciones_incorrectas))
    
    Args:
        state_dict: Diccionario con variables de estado
        all_actions: Lista de todas las acciones posibles
        reward_fn: Instancia de la función de recompensa
        agent_mock: Objeto mock del agente
        env_mock: Objeto mock del entorno
        state_tuple: Tupla de estado para pasar a reward_fn.compute()
        get_correct_action_fn: Función que determina la acción correcta
    
    Returns:
        Margen (float)
    """
    correct_action = get_correct_action_fn(state_dict)
    
    # Calcular recompensas para todas las acciones
    rewards = {}
    for action in all_actions:
        agent_mock.action = action
        try:
            rewards[action] = reward_fn.compute(agent_mock, env_mock, state_tuple)
        except Exception:
            rewards[action] = -1e6  # Penalización alta en caso de error
    
    # Obtener recompensa de la acción correcta
    reward_correct = rewards[correct_action]
    
    # Obtener la mejor recompensa de las acciones incorrectas
    incorrect_actions = [a for a in all_actions if a != correct_action]
    if incorrect_actions:
        reward_best_incorrect = max(rewards[a] for a in incorrect_actions)
    else:
        # Si solo hay una acción, el margen es 0
        reward_best_incorrect = reward_correct
    
    # Margen: diferencia entre correcta y mejor incorrecta
    margin = reward_correct - reward_best_incorrect
    
    return margin


def compute_total_margin(
    df,
    params: Dict,
    reward_class,
    state_columns: list,
    action_column: str,
    get_correct_action_fn: Callable,
    all_actions: list,
    env_extra_attrs: Dict = None,
    agent_extra_attrs: Dict = None,
    state_transform_fn: Callable = None
) -> float:
    """
    Calcula el margen total para un conjunto de hiperparámetros.
    
    Args:
        df: DataFrame con los datos de estado-acción
        params: Diccionario de hiperparámetros
        reward_class: Clase de la función de recompensa (ej: DefaultBatteryReward)
        state_columns: Lista de columnas que definen el estado
        action_column: Nombre de la columna de acción
        get_correct_action_fn: Función que determina la acción correcta
        all_actions: Lista de todas las acciones posibles
        env_extra_attrs: Atributos adicionales del entorno (opcional)
        agent_extra_attrs: Atributos adicionales del agente (opcional)
        state_transform_fn: Función para transformar el estado antes de pasarlo (opcional)
    
    Returns:
        Margen total (suma de márgenes de todos los estados únicos)
    """
    # Instanciar función de recompensa
    reward_fn = reward_class(**params)
    
    # Crear mocks de agente y entorno
    agent_mock = type('MockAgent', (), {})()
    env_mock = type('MockEnv', (), {})()
    
    # Asignar atributos extras si se proporcionan
    if env_extra_attrs:
        for key, value in env_extra_attrs.items():
            setattr(env_mock, key, value)
    
    if agent_extra_attrs:
        for key, value in agent_extra_attrs.items():
            setattr(agent_mock, key, value)
    
    # Agrupar por estado único
    df_states = df.groupby(state_columns, as_index=False).first()
    
    total_margin = 0.0
    
    for _, row in df_states.iterrows():
        # Construir diccionario de estado
        state_dict = {col: row[col] for col in state_columns}
        
        # Aplicar transformación si se proporciona
        if state_transform_fn:
            state_dict = state_transform_fn(state_dict, row)
        
        # Construir state_tuple según el agente
        state_tuple = tuple(row[col] for col in state_columns)
        
        # Actualizar env_mock con valores del estado si es necesario
        # (algunas funciones de recompensa leen del entorno)
        for key, value in state_dict.items():
            if not hasattr(env_mock, key):
                setattr(env_mock, key, value)
        
        # Calcular margen para este estado
        margin = compute_margin_for_state(
            state_dict=state_dict,
            all_actions=all_actions,
            reward_fn=reward_fn,
            agent_mock=agent_mock,
            env_mock=env_mock,
            state_tuple=state_tuple,
            get_correct_action_fn=get_correct_action_fn
        )
        
        total_margin += margin
    
    return total_margin


def compute_total_margin_from_csv(
    df,
    state_columns: list,
    action_column: str,
    reward_column: str,
    get_correct_action_fn: Callable,
    all_actions: list,
    state_transform_fn: Callable = None
) -> float:
    """
    Calcula el margen total usando las recompensas precalculadas del CSV.
    VERSIÓN OPTIMIZADA: No recalcula recompensas, usa valores del CSV.
    
    Args:
        df: DataFrame con columnas de estado, acción y recompensa
        state_columns: Lista de columnas que definen el estado
        action_column: Nombre de la columna de acción
        reward_column: Nombre de la columna de recompensa
        get_correct_action_fn: Función que determina la acción correcta
        all_actions: Lista de todas las acciones posibles
        state_transform_fn: Función para transformar el estado antes de pasarlo (opcional)
    
    Returns:
        Margen total (suma de márgenes de todos los estados únicos)
    """
    # Agrupar por estado único
    grouped = df.groupby(state_columns)
    
    total_margin = 0.0
    
    for state_values, group in grouped:
        # Construir diccionario de estado
        state_dict = dict(zip(state_columns, state_values))
        
        # Aplicar transformación si se proporciona
        if state_transform_fn:
            # Obtener la primera fila del grupo para pasar a transform
            first_row = group.iloc[0]
            state_dict = state_transform_fn(state_dict, first_row)
        
        # Construir state_tuple
        state_tuple = state_values if isinstance(state_values, tuple) else (state_values,)
        
        # Determinar la acción correcta
        correct_action = get_correct_action_fn(state_dict, state_tuple)
        
        # Crear diccionario acción -> recompensa para este estado
        rewards = {}
        for _, row in group.iterrows():
            action = row[action_column]
            reward = row[reward_column]
            rewards[action] = reward
        
        # Calcular margen
        if correct_action not in rewards:
            # Si no hay datos para la acción correcta, margen = 0
            continue
        
        reward_correct = rewards[correct_action]
        
        # Mejor recompensa de acciones incorrectas
        incorrect_actions = [a for a in all_actions if a != correct_action and a in rewards]
        if incorrect_actions:
            reward_best_incorrect = max(rewards[a] for a in incorrect_actions)
        else:
            # Si solo hay una acción, margen = 0
            reward_best_incorrect = reward_correct
        
        # Margen: diferencia entre correcta y mejor incorrecta
        margin = reward_correct - reward_best_incorrect
        total_margin += margin
    
    return total_margin


# ==================================================
# Diccionario de configuraciones por agente
# ==================================================

AGENT_CONFIGS = {
    'battery': {
        'get_correct_action': get_correct_action_battery,
        'all_actions': [0, 1, 2],
        'state_columns': ['battery_soc_idx', 'demand_power_idx', 'renewable_idx'],
        'extra_state_vars': ['delta_p_interno', 'soc_max'],
        'agent_attrs': {'soc_max': 4},
        'env_attrs': {'renewable_power_idx': 0, 'demand_power_idx': 0, 'price': 1.0},
    },
    'grid': {
        'get_correct_action': get_correct_action_grid,
        'all_actions': [0, 1],
        'state_columns': ['battery_soc_idx', 'demand_power_idx', 'renewable_idx'],
        'extra_state_vars': ['delta_p_interno'],
        'agent_attrs': {},
        'env_attrs': {'renewable_power_idx': 0, 'demand_power_idx': 0, 'price': 1.0},
    },
    'solar': {
        'get_correct_action': get_correct_action_solar,
        'all_actions': [0, 1],
        'state_columns': ['solar_power_idx', 'demand_power_idx', 'renewable_idx'],
        'extra_state_vars': ['solar_potential_idx'],
        'agent_attrs': {},
        'env_attrs': {'renewable_power_idx': 0, 'demand_power_idx': 0},
    },
    'wind': {
        'get_correct_action': get_correct_action_wind,
        'all_actions': [0, 1],
        'state_columns': ['wind_power_idx', 'demand_power_idx', 'renewable_idx'],
        'extra_state_vars': ['wind_potential_idx'],
        'agent_attrs': {},
        'env_attrs': {'renewable_power_idx': 0, 'demand_power_idx': 0},
    },
    'load': {
        'get_correct_action': get_correct_action_load,
        'all_actions': [0, 1],
        'state_columns': ['battery_soc_idx', 'demand_power_idx', 'renewable_idx', 'price'],
        'extra_state_vars': ['soc_idx', 'delta_p_interno', 'comfort_threshold'],
        'agent_attrs': {'comfort_threshold': 4},
        'env_attrs': {},
    },
}
