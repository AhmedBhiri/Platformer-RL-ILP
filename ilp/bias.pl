% Popper bias
head_pred(good_action,2).

% state feature predicates
body_pred(enemy_dist,2).
body_pred(gap_dist,2).
body_pred(on_ground,2).
body_pred(near,1).
body_pred(far,1).
body_pred(pit_near,1).
body_pred(enemy_near,1).

% action identity predicates (so Popper can talk about A)
body_pred(is_jump,1).
body_pred(is_do_nothing,1).
body_pred(is_attack,1).

% IMPORTANT: Popper expects 1-tuples with a trailing comma
type(good_action,(state,action)).
type(enemy_dist,(state,dist)).
type(gap_dist,(state,dist)).
type(on_ground,(state,bool)).
type(near,(dist,)).
type(far,(dist,)).
type(pit_near,(state,)).
type(enemy_near,(state,)).
type(is_jump,(action,)).
type(is_do_nothing,(action,)).
type(is_attack,(action,)).

direction(good_action,(in,in)).
direction(enemy_dist,(in,out)).
direction(gap_dist,(in,out)).
direction(on_ground,(in,out)).
direction(near,(in,)).
direction(far,(in,)).
direction(pit_near,(in,)).
direction(enemy_near,(in,)).
direction(is_jump,(in,)).
direction(is_do_nothing,(in,)).
direction(is_attack,(in,)).

% declare allowed action constants
action(do_nothing).
action(jump).
action(attack).

max_body(4).
max_vars(6).

