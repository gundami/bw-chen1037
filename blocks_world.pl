% Situation Calculus blocks world
%
% There is a single fluent on/3, and we will define clear/2 in terms of on/3.
% Because we will retract and assert the on/3 fluent, we need to declare that
% on/3 is dynamic
:- dynamic on/3.

% This is the initial state
% reset means that the blocks have been put in their initial configuration
reset:-
   retractall(on(_,_,[])),
   assert(on(a,1,[])),
   assert(on(b,3,[])),
   assert(on(c,a,[])).
   
% Compute all possible states of the blocks world
% state(State) means that State is a valid configuration of blocks
state(State):-
   (block(A);place(A)),dif(A,a),
   (block(B);place(B)),dif(B,b),
   (block(C);place(C)),dif(C,c),
   dif(A,B),
   dif(B,C),
   dif(A,C),
   grounded(A,B,C),
   atomics_to_string([A,B,C],State).

% grounded(A,B,C) means that a configuration of blocks is valid,
% where Block a is on A, Block b is on B, and Block c is on C.
grounded(A,B,C):-
   legal(A,B,C),
   (place(A);place(B);place(C)),!.

% legal(A,B,C) means there are no cycles in the configuration of blocks
% where Block a is on A, Block b is on B, and Block c is on C.
% A cycle is a case where (for example) a is on b and b is on a,
% or a is on b, b is on c, and c is on a.
legal(A,B,C):-
   block(A),block(B),A=b,B=a,!,fail
   ;
   block(B),block(C),B=c,C=b,!,fail
   ;
   block(A),block(C),A=c,C=a,!,fail
   ;
   true.

% current_state(State) means that State is the current state before any
% action has been performed
current_state(State):-
   on(a,A,[]),
   on(b,B,[]),
   on(c,C,[]),
   atomics_to_string([A,B,C],State).
   
% Step forward, performing action Act.  This is Non-Logical!
% step(Act) means that the current state is the new state resulting
% from performing Action Act.
step(Act):- poss([Act]),
   % first, determine the new positions of a, b, and c
   on(a,A,[Act]),
   on(b,B,[Act]),
   on(c,C,[Act]),
   % retract the "previous state"
   retractall(on(_,_,[])),
   % assert the new current state
   assert(on(a,A,[])),
   assert(on(b,B,[])),
   assert(on(c,C,[])).

% action(Act) means that Act is a well-formed but potentially impossible
% action.
action(Act):-
   Act = move(A,B,C),
   block(A),
   (block(B);place(B)),
   (block(C);place(C)),
   dif(A,B),
   dif(A,C),
   dif(B,C).
   
% There are three blocks, a, b, and c
block(a).
block(b).
block(c).

% There are four places, 1, 2, 3, and 4
place(1).
place(2).
place(3).
place(4).

% Initial state
% on(Block,Position,S) means that the Block is on Position in Situation S.
% In a Gymnasium environment, the reset predicate will override this
on(a,1,[]).
on(b,3,[]).
on(c,a,[]).

% Block X is on Y if it is moved onto Y
on(X,Y,[move(X,Z,Y)|S]):- poss([move(X,Z,Y)|S]).
% Block X is on Y after performing an action A if A is possible, and
% A does not involve moving X off of Y, and X was on Y in S
on(X,Y,[A|S]):-
  poss([A|S]),
  A \= move(X,Y,_),
  on(X,Y,S).

% clear(X,S) means block or position X is clear in Situation S.
clear(X,S):-
   on(_,X,S),!,fail
   ;
   true.

% poss([A|S]) means action A is possible in S
% poss([move(Block,From,To)|S]) means it is possible to move
% block Block from position From to position To, in S
poss([move(Block,From,To)|S]):-
  block(Block),
  clear(Block,S),
  (place(To);block(To)),
  dif(Block,From),
  dif(From,To),
  dif(Block,To),
  clear(To,S),
  (place(From);block(From)),
  on(Block,From,S).
