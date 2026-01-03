% Popper bias
head_pred(good_action,2).
body_pred(enemy_dist,2).
body_pred(gap_dist,2).
body_pred(on_ground,2).
body_pred(near,1).
body_pred(far,1).

type(good_action,(state,action)).
type(enemy_dist,(state,dist)).
type(gap_dist,(state,dist)).
type(on_ground,(state,bool)).
type(near,(dist)).
type(far,(dist)).

direction(good_action,(in,in)).
direction(enemy_dist,(in,out)).
direction(gap_dist,(in,out)).
direction(on_ground,(in,out)).
direction(near,(in)).
direction(far,(in)).

action(do_nothing).
action(jump).
action(attack).

max_body(4).
max_vars(6).

