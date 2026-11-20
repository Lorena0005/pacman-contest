# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent',
                second='DefensiveReflexAgent',
                num_training=0):
    """Make the two agents for our team."""
    agent_types = {
        'OffensiveReflexAgent': OffensiveReflexAgent,
        'DefensiveReflexAgent': DefensiveReflexAgent,
    }
    if first not in agent_types or second not in agent_types:
        raise ValueError('Unknown agent type in team options')
    return [agent_types[first](first_index), agent_types[second](second_index)]


class ReflexCaptureAgent(CaptureAgent):
    """Common tools used by both agents."""

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.home_entries = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)
        # Find the open squares next to the middle of our side.
        self.home_entries = self._find_home_entries(game_state)

    def _find_home_entries(self, game_state):
        walls = game_state.get_walls()
        midline = walls.width // 2
        # Red is on the left. Blue is on the right.
        home_x = range(midline) if self.red else range(midline, walls.width)
        home_cells = [
            (x, y)
            for x in home_x
            for y in range(walls.height)
            if not walls[x][y]
        ]
        if not home_cells:
            return [self.start] if self.start is not None else []
        boundary_x = midline - 1 if self.red else midline
        nearest_column = min(abs(x - boundary_x) for x, _ in home_cells)
        return [
            (x, y)
            for x, y in home_cells
            if abs(x - boundary_x) == nearest_column
        ]

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        position = successor.get_agent_state(self.index).get_position()
        if position != nearest_point(position):
            # Finish the move so the agent is on a full grid square.
            successor = successor.generate_successor(self.index, action)
        return successor

    def _closest_distance(self, position, targets):
        if position is None or not targets:
            # There is no target to measure.
            return float('inf')
        return min(self.get_maze_distance(position, target) for target in targets)

    def _nearest_target(self, position, targets):
        if position is None or not targets:
            return None
        return min(targets, key=lambda target: self.get_maze_distance(position, target))

    @staticmethod
    def _movement_penalty(game_state, agent_index, action):
        penalty = 0.0
        # Prefer moving forward instead of stopping or turning back.
        if action == Directions.STOP:
            penalty += 2.0
        direction = game_state.get_agent_state(agent_index).configuration.direction
        if action == Directions.REVERSE[direction]:
            penalty += 1.0
        return penalty

    def _home_distance(self, position):
        return self._closest_distance(position, self.home_entries)

    def choose_action(self, game_state):
        raise NotImplementedError


class OffensiveReflexAgent(ReflexCaptureAgent):
    """Get enemy food and return home before a ghost can catch us."""

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        current_state = game_state.get_agent_state(self.index)
        current_position = current_state.get_position()
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        opponents = [
            game_state.get_agent_state(index)
            for index in self.get_opponents(game_state)
        ]
        active_ghosts = [
            opponent.get_position()
            for opponent in opponents
            if (opponent.get_position() is not None
                and not opponent.is_pacman
                and opponent.scared_timer == 0)
        ]
        nearest_ghost = self._closest_distance(current_position, active_ghosts)
        carrying = current_state.num_carrying

        # Go home if we have enough food, there is little food left, or danger is close.
        returning_home = (
            current_state.is_pacman
            and carrying > 0
            and (carrying >= 5 or len(food) <= 2 or nearest_ghost <= 5)
        )
        food_target = self._nearest_target(current_position, food)
        capsule_target = self._nearest_target(current_position, capsules)
        nearest_capsule_distance = (
            self.get_maze_distance(current_position, capsule_target)
            if capsule_target is not None else float('inf')
        )
        nearest_food_distance = (
            self.get_maze_distance(current_position, food_target)
            if food_target is not None else float('inf')
        )
        seek_capsule = (
            not returning_home
            and nearest_ghost <= 8
            and nearest_capsule_distance <= nearest_food_distance + 3
        )

        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            next_state = successor.get_agent_state(self.index)
            next_position = next_state.get_position()
            value = 0.0

            # Move home with food, or move toward the closest food.
            if returning_home:
                value -= 3.0 * self._home_distance(next_position)
            else:
                remaining_food = self.get_food(successor).as_list()
                if remaining_food:
                    value -= 2.0 * self._closest_distance(
                        next_position, remaining_food
                    )
                else:
                    value -= 2.0 * self._home_distance(next_position)

                food_eaten = len(food) - len(remaining_food)
                value += 10.0 * food_eaten

                # A capsule can help when an active ghost is nearby.
                if seek_capsule and capsule_target is not None:
                    remaining_capsules = self.get_capsules(successor)
                    if capsule_target in remaining_capsules:
                        value -= 1.5 * self.get_maze_distance(
                            next_position, capsule_target
                        )
                    else:
                        value += 8.0

            if next_state.is_pacman:
                threat_distance = self._closest_distance(
                    next_position, active_ghosts
                )
                if threat_distance <= 5:
                    # Be extra careful when carrying food near a ghost.
                    value -= (6 - threat_distance) * (
                        12.0 + 2.0 * next_state.num_carrying
                    )
            elif next_state.scared_timer > 0:
                visible_invaders = [
                    opponent.get_position()
                    for opponent in opponents
                    if opponent.is_pacman and opponent.get_position() is not None
                ]
                invader_distance = self._closest_distance(
                    next_position, visible_invaders
                )
                if invader_distance <= 3:
                    value -= (4 - invader_distance) * 8.0

            value -= self._movement_penalty(game_state, self.index, action)
            values.append(value)

        best_value = max(values)
        return random.choice([
            action for action, value in zip(actions, values)
            if value == best_value
        ])


class DefensiveReflexAgent(ReflexCaptureAgent):
    """Chase enemy Pacmen and guard our side."""

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        agent_state = game_state.get_agent_state(self.index)
        position = agent_state.get_position()
        opponents = [
            game_state.get_agent_state(index)
            for index in self.get_opponents(game_state)
        ]
        invaders = [
            opponent.get_position()
            for opponent in opponents
            if opponent.is_pacman and opponent.get_position() is not None
        ]

        target = None
        chase_invader = bool(invaders) and agent_state.scared_timer == 0
        if agent_state.is_pacman:
            # If we crossed the middle, return to our own side.
            target = self._nearest_target(position, self.home_entries)
        elif chase_invader:
            target = self._nearest_target(position, invaders)
        else:
            previous = self.get_previous_observation()
            if previous is not None:
                # Check if food disappeared since our last turn.
                previous_food = set(
                    self.get_food_you_are_defending(previous).as_list()
                )
                current_food = set(
                    self.get_food_you_are_defending(game_state).as_list()
                )
                missing_food = list(previous_food - current_food)
                target = self._nearest_target(position, missing_food)
            if target is None:
                # No lost food or invader was seen, so guard the middle.
                target = self._nearest_target(position, self.home_entries)

        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            next_state = successor.get_agent_state(self.index)
            next_position = next_state.get_position()
            if next_state.is_pacman:
                value = -3.0 * self._home_distance(next_position)
            elif target is not None:
                distance = self.get_maze_distance(next_position, target)
                value = -2.0 * distance
                if chase_invader:
                    # Give a bonus for getting closer to an invader.
                    value += 5.0 * (self._closest_distance(position, invaders) - distance)
            else:
                value = 0.0

            if invaders and agent_state.scared_timer > 0:
                # Keep away from invaders while we are scared.
                distance = self._closest_distance(next_position, invaders)
                value += min(distance, 8) * 1.5

            value -= self._movement_penalty(game_state, self.index, action)
            values.append(value)

        best_value = max(values)
        return random.choice([
            action for action, value in zip(actions, values)
            if value == best_value
        ])
