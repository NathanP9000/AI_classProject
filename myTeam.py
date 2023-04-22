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
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'offenseAgent', second = 'defenseAgent'):
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

  # successor gameState -> find our agents position getCapsules return true
  def isGoal(self,successor): #Gamestate
    previousState = self.getPreviousObservation()
    if ((not successor.getAgentState(self.index).isPacman)) : #Need to find specific agent id
      return True
    return False

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index) # south north ease est stop
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    
    values = [self.evaluate(gameState, a) for a in actions] # Given legal actions. Evaluate each of them [10,30,-1300,1,3]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    #print(values)
    if not gameState.getAgentState(self.index).isPacman:
      self.food = len(self.getFood(gameState).asList())
    #Added code for detecting someone close to the pacman
    myPos = gameState.getAgentState(self.index).getPosition()

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if ((not a.isPacman) and a.getPosition() != None)]
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
              dist = self.getMazeDistance(previousPosition, currentPosition)
              if dist > 1:
                continue
              stack.push(child)
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        if solution == []:
          return  random.choice(actions)
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
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







class defenseAgent(CaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    actions = gameState.getLegalActions(self.index) # south north ease est stop
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    values = []

    # If no defender exists then do evaluate 2
    if(len(invaders) > 0):
      values = [self.evaluate(gameState, a) for a in actions]
    else:
      values = [self.evaluate2(gameState,a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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

  def getFeatures2(self, gameState, action):
    #Useful variables
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFoodYouAreDefending(successor).asList()    
    enemies = self.getOpponents(successor) # enemies indices in list
    

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      mF=0
      mF2=0
      minDistance = 1000
      minDistance2 = 1000
      for food in foodList:
        dist1 = self.getMazeDistance(successor.getAgentState(enemies[0]).getPosition(), food)
        dist2 = self.getMazeDistance(successor.getAgentState(enemies[1]).getPosition(), food) 
        if minDistance >  dist1:
          minDistance = dist1
          mF = food
        if minDistance2 > dist2:
          minDistance2 = dist2
          mF2 = food
      
    if minDistance > minDistance2:
      features['distanceToFood'] = self.getMazeDistance(myPos, mF2)
    else:
     features['distanceToFood'] = self.getMazeDistance(myPos, mF2)


    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    
    # Computes distance to enemy ghosts
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if ((not a.isPacman) and a.getPosition() != None)]
    features['notInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['enemyDistance'] = abs(min(dists))

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    return features

  def getWeights2(self, gameState, action):
    return {'numInvaders': -1000,'distanceToFood':-1 ,'onDefense': 1000, 'enemyDistance': -1, 'stop': -100, 'reverse': -2}


