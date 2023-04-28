# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from util import PriorityQueue
import random
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'offenseAgent', second = 'mirrorDefenseAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class offenseAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.food = len(self.getFood(gameState).asList()) # holds number of food before crossing over
    self.stuck_counter = 0 # initialize stuck counter to 0
    self.last_position = (0,0)
    self.actionsToDo = []

  # successor gameState -> find our agents position getCapsules return true
  def isGoal(self,successor): #Gamestate
    previousState = self.getPreviousObservation()
    capsulePath = False    
    if successor.getAgentState(self.index).getPosition() in previousState.getCapsules():
      capsulePath = True
    if ((not successor.getAgentState(self.index).isPacman) or capsulePath) : #Need to find specific agent id
      self.debugDraw(successor.getAgentState(self.index).getPosition(), (200,200, 200))  
      self.debugClear()
      return True
    return False

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    myPos = gameState.getAgentState(self.index).getPosition()
    # increment stuck counter if agent is not moving
    if myPos[1] == self.last_position[1]:
        self.stuck_counter += 1
    else:
        self.stuck_counter = 0
    self.last_position = myPos

    # reposition to topside or bottomside if stuck for 10-20 moves
    if self.stuck_counter >= 15 and not gameState.getAgentState(self.index).isPacman:
      entrances = self.findEntrances(gameState)

      if gameState.isOnRedTeam(self.index):
          target_pos = (15, random.choice(entrances))
      else:
          target_pos = (16, random.choice(entrances))
      # go to new area
      self.actionsToDo = self.aStarSearch(gameState, myPos, target_pos)
    if len(self.actionsToDo) > 0:
      return self.actionsToDo.pop(0)
    
    actions = gameState.getLegalActions(self.index) # south north ease est stop
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    values = [self.evaluate(gameState, a) for a in actions] # Given legal actions. Evaluate each of them [10,30,-1300,1,3]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    #print(values)
    if not gameState.getAgentState(self.index).isPacman:
      self.food = len(self.getFood(gameState).asList())
    #Added code for detecting someone close to the pacman

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if ((not a.isPacman) and (not a.scaredTimer > 10) and a.getPosition() != None)]
    #print("self",self.index)
    if len(defenders)> 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      closest = min(dists)
      if ( not gameState.getAgentState(self.index).isPacman and  closest < 4) or ( gameState.getAgentState(self.index).isPacman and  closest < 5): # If ghost is near u find shortest path back using bfs
        solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
        stack = util.Queue()
        counter = util.Counter()
        if self.isGoal(gameState):
          return Directions.STOP
        for a in actions:
          stack.push((gameState.generateSuccessor(self.index,a),a,a))
        while stack.isEmpty() is not True:
            node = stack.pop() # stack pops a state and action list and visited list
            if counter[node[0].getAgentState(self.index).getPosition()] == 1:
                continue
            #print(node[0].getAgentState(self.index).getPosition())
            if self.isGoal(node[0]):
                solution = node[1]
                break
            counter[node[0].getAgentState(self.index).getPosition()] = 1
            children = []  
            for a in node[0].getLegalActions(self.index):
              children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
            for child in children:
              previousPosition = node[0].getAgentState(self.index).getPosition()
              currentPosition = child[0].getAgentState(self.index).getPosition()
              self.debugDraw(previousPosition, (100,100, 100)) 
              dist = self.getMazeDistance(previousPosition, currentPosition)
              if dist > 1:
                continue
              stack.push(child)
        if solution == []:
          return  random.choice(actions)
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        return solution

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue] # Choose the best action
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:#Can food ever be less than 2?
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions) #given 2 or more moves that have the same heuristic value choose one randomly

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #if self.index is 1:
      #print(features)
      #print(weights)
      #print(features*weights) # the result is a single integer that is computed by multiplying out the common keys
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()  
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    #print(features)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100,'distanceToFood': -1}
  
  def aStarSearch(self, gameState, currentPos, targetPos):
    # Define heuristic function
    def heuristic(state):
        #print("manhattan distance: ", util.manhattanDistance(state.getAgentState(self.index).getPosition(), targetPos) )
        return util.manhattanDistance(state.getAgentState(self.index).getPosition(), targetPos) 

    # Initialize priority queue with start state
    pq = PriorityQueue()
    pq.push((gameState, [], heuristic(gameState)), heuristic(gameState))

    # Initialize visited set
    visited = set()

    # A* search loop
    while not pq.isEmpty():
        state, actions, cost = pq.pop()
        currentPos = state.getAgentState(self.index).getPosition()

        if currentPos == targetPos:
            return actions

        if state.getAgentState(self.index).isPacman:
          continue

        if currentPos in visited:
            continue

        visited.add(currentPos)

        for action in state.getLegalActions(self.index):
            successor = state.generateSuccessor(self.index, action)
            successorState = (successor, actions + [action], cost + heuristic(successor)+1)
            pq.push(successorState, cost + heuristic(successor)+1)
    
    # If no winning action found, return random legal action
    #print("Tried to route from ", currentPos, " to ", targetPos, " and failed.")
    return [random.choice(state.getLegalActions(self.index))]

  # returns a list of entrances
  def findEntrances(self, gameState):
    """
    Returns a list of entrances between the two horizontal sides of the board.
    """
    walls = gameState.getWalls()
    height = gameState.data.layout.height
    entrances = []



    for y in range(1, height - 1):
        if not walls[15][y] and not walls[16][y]:
            if len(entrances) == 0 or y > entrances[-1] + 1:
                entrances.append(y)

    return entrances

# perfect mirror
class mirrorDefenseAgent(CaptureAgent):
  """
  First chooses a random enemy agent to mirror.
  If the enemy agent turns into a pac man, that pacman is listed as the offense agent and is mirrored now
  """
  def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      if self.red:
        self.ourSideCenter = gameState.data.layout.width // 2 - 1
        self.theirSideCenter = gameState.data.layout.width // 2
        self.enemyOffenseAgent = 1
      else:
        self.enemyOffenseAgent = 0
        self.ourSideCenter = gameState.data.layout.width // 2 
        self.theirSideCenter = gameState.data.layout.width // 2 - 1
      self.entrances = self.findEntrances(gameState)
      self.enemyClosestEntrance = self.entrances[0]
      # our side center and their side center for entrance calculations
      print("Entrances: ", self.entrances)
      for entrance in self.entrances:
        self.debugDraw((self.ourSideCenter, entrance), (200, 200, 200))
        self.debugDraw((self.theirSideCenter, entrance), (100, 100, 200))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index) # south north ease est stop
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemiesIndexes = [i for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = []
    if len(invaders) > 0:

      # ensure you have the right enemy offense agent
      # find the closer invader
      #minDistance = 100
      #closestInvader = -1
      #for invader in invaders:
      #  if self.distancer.getDistance(invader.getPosition(), gameState.getPosition()) < minDistance:
      #    minDistance = self.distancer.getDistance(invader.getPosition(), gameState.getPosition())
      #    closestInvader = invader.index        
      #self.enemyOffenseAgent = invaders[closestInvader]
      self.enemyOffenseAgent = enemiesIndexes[0]
      values = [self.evaluate(gameState, a) for a in actions]
    else: #no attackers, mirror enemy
      # determine which entrance the enemy is closest to
      minDistance = 100
      self.enemyClosestEntrance = self.entrances[0]
      for entry in self.entrances:
        if self.getMazeDistance(gameState.getAgentState(self.enemyOffenseAgent).getPosition(), (self.theirSideCenter,entry)) < minDistance:
          minDistance = self.getMazeDistance(gameState.getAgentState(self.enemyOffenseAgent).getPosition(), (self.theirSideCenter,entry))
          #print("Min distance: ", minDistance)
          self.enemyClosestEntrance = entry
      self.debugDraw((self.theirSideCenter, self.enemyClosestEntrance), (50, 50, 50))
      #print("Closest entrance is at y-level: ", self.enemyClosestEntrance)

      #bfs
      return self.bfs(gameState, actions)

      values = [self.evaluate2(gameState, a) for a in actions] # Given legal actions. Evaluate each of them [10,30,-1300,1,3]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    #print(values)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue] # Choose the best action
    foodLeft = len(self.getFood(gameState).asList()) 

    # printing the actions and their values
    print("values: ", values)
    print("actions: ", actions)
        
    
    if foodLeft <= 2:#Can food ever be less than 2?
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions) #given 2 or more moves that have the same heuristic value choose one randomly

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    #print(features)
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def evaluate2(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures2(gameState, action)
    weights = self.getWeights2(gameState, action)
    return features * weights

  #Stay near food close to enemy 
  def getFeatures2(self, gameState, action):
    #Useful variables
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFoodYouAreDefending(successor).asList()    
    enemies = self.getOpponents(successor) # enemies indices in list
    

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # distance to their closest entrance
    features['distanceToEnemyEntrance'] = self.getMazeDistance(successor.getAgentState(self.index).getPosition(), (self.ourSideCenter, self.enemyClosestEntrance))
    

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    return features

  def getWeights2(self, gameState, action):
    return {'onDefense': 0, 'distanceToEnemyEntrance': -1}

  # returns a list of entrances
  def findEntrances(self, gameState):
    """
    Returns a list of entrances between the two horizontal sides of the board.
    """
    walls = gameState.getWalls()
    height = gameState.data.layout.height
    entrances = []



    for y in range(1, height - 1):
        if not walls[self.ourSideCenter][y] and not walls[self.theirSideCenter][y]:
          entrances.append(y)

    return entrances
  def bfs(self,gameState,actions):
    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.Queue()
    counter = util.Counter()
    if self.isGoal(gameState):
      return Directions.STOP
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a,a))
    while stack.isEmpty() is not True:
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        if node[0].getAgentState(self.index).isPacman:
          continue
        #print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal(node[0]):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
        for child in children:
          stack.push(child)
    if solution == []:
      self.offense = False
      return  random.choice(actions)
    #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    self.offense = False
    return solution

  # goal state is when you are mirroring their offense agent on their closest entrance
  def isGoal(self, gameState):
    previousState = self.getPreviousObservation()
    if gameState.getAgentState(self.index).getPosition() == (self.ourSideCenter, self.enemyClosestEntrance): #goal is to match them
      return True
    return False

  def bfs2(self,gameState,actions):
    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.Queue()
    counter = util.Counter()
    if self.isGoal(gameState):
      return Directions.STOP
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a,a))
    while stack.isEmpty() is not True:
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        if node[0].getAgentState(self.index).isPacman:
          continue
        #print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal(node[0]):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
        for child in children:
          stack.push(child)
    if solution == []:
      self.offense = False
      return  random.choice(actions)
    #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    self.offense = False
    return solution

  # goal state is when you are mirroring their offense agent on their closest entrance
  def isGoal2(self, gameState):
    previousState = self.getPreviousObservation()
    if gameState.getAgentState(self.index).getPosition() == (self.ourSideCenter, self.enemyClosestEntrance): #goal is to match them
      return True
    return False
