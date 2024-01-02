# 바닥부터 배우는 강화 학습 P.139 Monte Carlo Control 구현

import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.first_block = range(3)
        self.second_block = range(2, 5)
    
    # Agent의 움직임을 나타냄
    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()
        
        state = self.get_state()
        reward = -1
        done = self.is_done()

        return state, reward, done
    
    # 회색인 부분으로는 갈 수 없음
    def move_left(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y in self.first_block:
            pass
        elif self.x == 5 and self.y in self.second_block:
            pass
        else:
            self.x -= 1
    
    def move_right(self):
        if self.x == 6:
            pass
        elif self.x == 1 and self.y in self.first_block:
            pass
        elif self.x == 3 and self.y in self.second_block:
            pass
        else:
            self.x += 1
    
    # y축 방향이 반대임
    def move_up(self):
        if self.y == 0:
            pass
        elif self.x == 2 and self.y == 3:
            pass
        else:
            self.y -= 1
    
    def move_down(self):
        if self.y == 4:
            pass
        elif self.x == 4 and self.y == 1:
            pass
        else:
            self.y += 1

    # 종료 State에 도달했는지 체크
    def is_done(self):
        if self.x == 6 and self.y == 4:
            return True
        else:
            return False
    
    # 현재 Agent가 위치한 State를 반환
    def get_state(self):
        return (self.x, self.y)
    
    # 종료 State에 도달했을 때 리셋
    def reset(self):
        self.x = 0
        self.y = 0

class QAgent(): # Action Value를 보고 Greedy하게 움직이는 Agent
    def __init__(self):
        self.q_table = np.zeros((7, 5, 4)) # Action Value를 저장하는 변수. (x, y, action_value)로 구성됨. 모두 0으로 초기화
        self.eps = 0.9 # epsilon. epsilon-greedy에서 사용함
        self.alpha = 0.01

    def select_action(self, state):
        # epsilon-greedy로 Action을 선택해 준다.
        x, y = state
        coin = random.random()
        if coin < self.eps: # 랜덤하게 선택하는 경우, decaying epsilon-greedy를 사용하므로 점차 epsilon이 낮아짐
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action
    
    def update_table(self, history):
        # 한 Episode에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트한다.
        cum_reward = 0 # Return
        for transition in history[::-1]: #history를 뒤에서부터 탐색하므로 ::-1
            state, action, reward, s_prime = transition
            x, y = state
            # MC 방식으로 Evaluation, value = value + a(Return-value)
            self.q_table[x,y,action] = self.q_table[x,y,action] + self.alpha * (cum_reward - self.q_table[x,y,action])
            cum_reward = cum_reward + reward # discount factor가 1이기 때문

    # epsilon을 줄여주는 함수
    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)
    
    # q_table에서 State 별로 Action Value가 가장 높은 Action을 보여줌
    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((7, 5))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data.T)

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 총 1,000 Episode 동안 학습
        done = False
        history = []

        env.reset()
        state = env.get_state()
        while not done:
            
            action = agent.select_action(state)
            state_prime, reward, done = env.step(action)
            history.append((state, action, reward, state_prime))
            state = state_prime
        agent.update_table(history) # 한 Epiosde가 끝났으면 history를 이용하여 업데이트
        agent.anneal_eps() # epsilon 값을 줄여줌

    agent.show_table() # 학습이 끝나고 결과를 출력

main()