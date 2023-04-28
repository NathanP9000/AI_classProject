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
    if self.red:
      self.ourSideCenter = gameState.data.layout.width // 2 - 1
      self.theirSideCenter = gameState.data.layout.width // 2
      self.enemyOffenseAgent = 1
    else:
      self.enemyOffenseAgent = 0
      self.ourSideCenter = gameState.data.layout.width // 2 
      self.theirSideCenter = gameState.data.layout.width // 2 - 1
    self.entrances = self.findEntrances(gameState)
    self.up = True
    self.targetCap = (0,0)
    self.capsuleTarget = False 

  # successor gameState -> find our agents position getCapsules return true
  # def isGoal(self,successor): #Gamestate
  #   previousState = self.getPreviousObservation()
  #   capsulePath = False    
  #   if successor.getAgentState(self.index).getPosition() in previousState.getCapsules():
  #     capsulePath = True
  #   if (capsulePath) : #Need to find specific agent id
  #     return True
  #   return False
  # def bfs(self, gameState, actions):
  #   solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
  #   stack = util.Queue()
  #   counter = util.Counter()
  #   if self.isGoal(gameState):
  #     return Directions.STOP
  #   for a in actions:
  #     stack.push((gameState.generateSuccessor(self.index,a),a,a))
  #   while stack.isEmpty() is not True:
  #       node = stack.pop() # stack pops a state and action list and visited list
  #       if counter[node[0].getAgentState(self.index).getPosition()] == 1:
  #           continue
  #       if node[0].getAgentState(self.index).isPacman:
  #         continue
  #       ##print(node[0].getAgentState(self.index).getPosition())
  #       if self.isGoal(node[0]):
  #           solution = node[1]
  #           break
  #       counter[node[0].getAgentState(self.index).getPosition()] = 1
  #       children = []  
  #       for a in node[0].getLegalActions(self.index):
  #         children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
  #       for child in children:
  #         stack.push(child)
  #   if solution == []:
  #     #print("random choice")
  #     return  random.choice(actions)
  #   ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
  #   return solution


  #Bfs to up and don
  def isGoal2(self,successor, targetPosition):
    if successor.getAgentState(self.index).getPosition() == targetPosition:
      return True
    return False

  def bfs2(self, gameState, actions, targetPos):
    def heuristic(gameState2):
      print("Heuristic ,",self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition()))
      return self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition())

    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.PriorityQueue()
    counter = util.Counter()
    if self.isGoal2(gameState, targetPos):
      return Directions.STOP
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a,a),heuristic(gameState.generateSuccessor(self.index,a)))
    while stack.isEmpty() is not True:

        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        ##print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal2(node[0],targetPos):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []  
        self.debugDraw(node[0].getAgentPosition(self.index), (200,100,150))
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
        for child in children:
          stack.push(child,heuristic(child[0]))
    if solution == []:
      return  (random.choice(actions), False)
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return (solution, True)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.debugClear()
    actions = gameState.getLegalActions(self.index)
    if self.targetCap not in self.getCapsules(gameState):
      self.capsuleTarget = False
    if self.capsuleTarget == True:
      ret =  self.bfs2(gameState,actions,self.targetCap)
      self.debugDraw(self.targetCap, (120,120,200))
      if(ret[1]== False):
        print("Abort")
        self.capsuleTarget = False
      else:
        return ret[0]
    myPos = gameState.getAgentState(self.index).getPosition()

    actions = gameState.getLegalActions(self.index) # south north ease est stop

    entrances = self.findEntrances(gameState)
    curr = myPos[1]
    chosen=-1
    act = Directions.STOP
    # if self.red and myPos[0] > 13:
    #   act = self.bfs(gameState, actions) 
    # elif not self.red and myPos[0] < 20:
    #   act = self.bfs(gameState, actions) 
    for cap in self.getCapsules(gameState):
      enemyDist = min([self.distancer.getDistance(cap, gameState.getAgentState(a).getPosition()) for a in self.getOpponents(gameState)])
      print(self.getCapsules(gameState))
      myDist = self.distancer.getDistance(cap, gameState.getAgentState(self.index).getPosition())
      print(enemyDist, myDist)
      if not self.capsuleTarget and  myDist< enemyDist:
        self.capsuleTarget = True
        self.targetCap = cap
        print(self.targetCap)
    if self.capsuleTarget:
      print("bfs", self.targetCap)
      ret =  self.bfs2(gameState,actions,self.targetCap)
      if(ret[1]== False):
        print("Abort")
        self.capsuleTarget = False
      else:     
        return ret[0]
      

    if act != Directions.STOP:
      print(act)
      return act
    if not self.up:
      start = len(entrances)-1
      while start >=0 :
        if entrances[start] < curr:
          chosen = entrances[start]
          break
        start-=1
      if start == 0: 
        self.up = True
    else:
      start = 0
      while start <len(entrances) :
        if entrances[start] > curr:
          chosen = entrances[start]
          break
        start+=1   
      if start  >= len(entrances)-1: 
        self.up = False
    target_pos = (self.ourSideCenter, chosen)
    #print(target_pos)
    #self.debugDraw(target_pos, (100,100,100))
    ret = self.bfs2(gameState,actions,target_pos)
    return ret[0]

    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue] # Choose the best action
    # foodLeft = len(self.getFood(gameState).asList())

    # if foodLeft <= 2:#Can food ever be less than 2?
    #   bestDist = 9999
    #   for action in actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    # return random.choice(bestActions) #given 2 or more moves that have the same heuristic value choose one randomly


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
  def evaluate2(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures2(gameState, action)
    weights = self.getWeights2(gameState, action)
    return features * weights

  def getFeatures2(self, gameState,action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    cap = self.getCapsules(gameState)
    # Compute distance to the nearest food
    if len(cap) > 0: # If there exist capsules
      myPos = successor.getAgentState(self.index).getPosition()  
      minDistance = min([self.getMazeDistance(myPos, caps) for caps in cap])
      features['distanceToCap'] = minDistance
    ##print(features)
    return features

  def getWeights2(self, gameState, action):
    return {'successorScore': 100,'distanceToCap': -1}
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #if self.index is 1:
      ##print(features)
      ##print(weights)
      ##print(features*weights) # the result is a single integer that is computed by multiplying out the common keys
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
    ##print(features)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100,'distanceToFood': -1}
  

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
      ##print("Entrances: ", self.entrances)
      # for entrance in self.entrances:
      #   self.debugDraw((self.ourSideCenter, entrance), (200, 200, 200))
      #   self.debugDraw((self.theirSideCenter, entrance), (100, 100, 200))

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
          ##print("Min distance: ", minDistance)
          self.enemyClosestEntrance = entry
      self.debugDraw((self.theirSideCenter, self.enemyClosestEntrance), (50, 50, 50))
      ##print("Closest entrance is at y-level: ", self.enemyClosestEntrance)

      #bfs
      return self.bfs(gameState, actions)

      values = [self.evaluate2(gameState, a) for a in actions] # Given legal actions. Evaluate each of them [10,30,-1300,1,3]
    # #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    ##print(values)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue] # Choose the best action
    foodLeft = len(self.getFood(gameState).asList()) 

    # #printing the actions and their values
    #print("values: ", values)
    #print("actions: ", actions)
        
    
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

    ##print(features)
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
        ##print(node[0].getAgentState(self.index).getPosition())
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
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
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
        ##print(node[0].getAgentState(self.index).getPosition())
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
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    self.offense = False
    return solution

  # goal state is when you are mirroring their offense agent on their closest entrance
  def isGoal2(self, gameState):
    previousState = self.getPreviousObservation()
    if gameState.getAgentState(self.index).getPosition() == (self.ourSideCenter, self.enemyClosestEntrance): #goal is to match them
      return True
    return False
