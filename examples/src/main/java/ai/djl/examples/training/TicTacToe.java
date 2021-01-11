/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.examples.training;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.util.Arrays;

/** A simple {@link RlEnv} for playing TicTacToe. */
public class TicTacToe implements RlEnv {

    private NDManager manager;
    private State state;
    private ReplayBuffer replayBuffer;

    /**
     * Constructs a {@link TicTacToe} with a basic {@link LruReplayBuffer}.
     *
     * @param manager the manager for creating the game in
     * @param batchSize the number of steps to train on per batch
     * @param replayBufferSize the number of steps to hold in the buffer
     */
    public TicTacToe(NDManager manager, int batchSize, int replayBufferSize) {
        this(manager, new LruReplayBuffer(batchSize, replayBufferSize));
    }

    /**
     * Constructs a {@link TicTacToe}.
     *
     * @param manager the manager for creating the game in
     * @param replayBuffer the replay buffer for storing data
     */
    public TicTacToe(NDManager manager, ReplayBuffer replayBuffer) {
        this.manager = manager;
        this.state = new State(new int[9], 1);
        this.replayBuffer = replayBuffer;
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        Arrays.fill(state.board, 0);
        state.turn = 1;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }

    /** {@inheritDoc} */
    @Override
    public NDList getObservation() {
        return state.getObservation(manager);
    }

    /** {@inheritDoc} */
    @Override
    public ActionSpace getActionSpace() {
        return state.getActionSpace(manager);
    }

    /** {@inheritDoc} */
    @Override
    public Step step(NDList action, boolean training) {
        int move = action.singletonOrThrow().getInt();
        if (move < 0 || move >= 9) {
            throw new IllegalArgumentException("Your move is out of bounds");
        }
        if (state.board[move] != 0) {
            throw new IllegalArgumentException("Your move is on an already occupied space");
        }
        State preState = state;

        state = new State(preState.board.clone(), -preState.turn);
        state.board[move] = preState.turn;

        TicTacToeStep step = new TicTacToeStep(manager.newSubManager(), preState, state, action);
        if (training) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    /** {@inheritDoc} */
    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return state.toString();
    }

    /** The {@link ai.djl.modality.rl.env.RlEnv.Step} for {@link TicTacToe}. */
    static final class TicTacToeStep implements RlEnv.Step {

        private NDManager manager;
        private State preState;
        private State postState;
        private NDList action;

        private TicTacToeStep(NDManager manager, State preState, State postState, NDList action) {
            this.manager = manager;
            this.preState = preState;
            this.postState = postState;
            this.action = action;
        }

        /** {@inheritDoc} */
        @Override
        public NDList getPreObservation() {
            return preState.getObservation(manager);
        }

        /** {@inheritDoc} */
        @Override
        public NDList getAction() {
            return action;
        }

        /** {@inheritDoc} */
        @Override
        public NDList getPostObservation() {
            return postState.getObservation(manager);
        }

        /** {@inheritDoc} */
        @Override
        public ActionSpace getPostActionSpace() {
            return postState.getActionSpace(manager);
        }

        /** {@inheritDoc} */
        @Override
        public NDArray getReward() {
            return manager.create((float) postState.getWinner());
        }

        /** {@inheritDoc} */
        @Override
        public boolean isDone() {
            return postState.getWinner() != 0 || postState.isDraw();
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            manager.close();
        }
    }

    /** A helper to manage the state of the game at a moment in time. */
    private static final class State {

        int[] board;
        int turn;

        private State(int[] board, int turn) {
            this.board = board;
            this.turn = turn;
        }

        private NDList getObservation(NDManager manager) {
            return new NDList(manager.create(board), manager.create(turn));
        }

        private ActionSpace getActionSpace(NDManager manager) {
            ActionSpace actionSpace = new ActionSpace();
            for (int i = 0; i < 9; i++) {
                if (board[i] == 0) {
                    actionSpace.add(new NDList(manager.create(i)));
                }
            }
            return actionSpace;
        }

        private boolean isDraw() {
            for (int i = 0; i < 9; i++) {
                if (board[i] == 0) {
                    return false;
                }
            }
            return true;
        }

        private int getWinner() {
            // win vertically
            for (int i = 0; i < 3; i++) {
                if (board[i] != 0 && board[i] == board[i + 3] && board[i] == board[i + 6]) {
                    return board[i];
                }
            }
            // win horizontally
            for (int i = 0; i < 9; i += 3) {
                if (board[i] != 0 && board[i] == board[i + 1] && board[i] == board[i + 2]) {
                    return board[i];
                }
            }

            // win descending diagonal
            if (board[0] != 0 && board[0] == board[4] && board[0] == board[8]) {
                return board[0];
            }

            // win ascending diagonal
            if (board[2] != 0 && board[2] == board[4] && board[2] == board[6]) {
                return board[2];
            }

            // no winner
            return 0;
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            StringBuilder s = new StringBuilder();
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    int ind = 3 * row + col;
                    int val = board[ind];
                    if (val == 0) {
                        s.append(' ');
                    } else if (val == 1) {
                        s.append('X');
                    } else {
                        s.append('O');
                    }
                    if (col < 2) {
                        s.append('|');
                    } else {
                        s.append('\n');
                    }
                }
                if (row < 2) {
                    s.append("-----\n");
                }
            }
            if (getWinner() != 0) {
                s.append("Winner: ").append(getWinner()).append('\n');
            }
            return s.toString();
        }
    }
}
