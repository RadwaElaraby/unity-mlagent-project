using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.Linq;

public class MyAgent : MonoBehaviour
{
    public float rotateSpeed;
    public float forceSpeed;

    [HideInInspector]
    public Tile onTile;
    [HideInInspector]
    public int id;
    [HideInInspector]
    public int order;
    [HideInInspector]
    public int target_tile_id;
    [HideInInspector]
    public bool reached_goal;
    [HideInInspector]
    public bool _reached_goal;
    [HideInInspector]
    public bool CollidedWithWall;
    [HideInInspector]
    public bool CollidedWithAgent;

    AreaManager areaManager;
    AgentHelper agentHelper;

    Settings settings;

    public float timeBetweenDecisionsAtInference;

    public void Awake()
    {
        areaManager = GetComponentInParent<AreaManager>();
        settings = areaManager.getSettings();
        agentHelper = new AgentHelper(settings);
    }

    public void CollectObservations(VectorSensor sensor)
    {
        var rangeRowStart = onTile.row - 1;
        var rangeRowEnd = onTile.row + 1;
        var rangeColStart = onTile.col - 1;
        var rangeColEnd = onTile.col + 1;


        // observation space
        // (((7*7)*4 + 2 + 2 ) * 2) + 2*2
        // (((7*7)*3 + n + 2 ) * n) + n*n
        // NUM_AGENTS n

        // all walls 49
        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                if (settings.WALLS[row, col] != null)
                    sensor.AddObservation(1);
                else
                    sensor.AddObservation(0);
            }
        }

        // all agents 49
        /*for (int row = 0; row < settings.mapHeight; row++)
        {
            for (int col = 0; col < settings.mapWidth; col++)
            {
                if (settings.agents[row, col] != null && !(row == onTile.row && col == onTile.col))
                    sensor.AddObservation(1);
                else
                    sensor.AddObservation(0);
            }
        }*/

        // me 49
        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                if (row == onTile.row && col == onTile.col)
                    sensor.AddObservation(1);
                else
                    sensor.AddObservation(0);
            }
        }

        /*
        // all targets 6
        for (int row = 0; row < settings.mapHeight; row++)
        {
            for (int col = 0; col < settings.mapWidth; col++)
            {
                if (Array.IndexOf(Settings.TARGET_TILES_IDs, settings.map[row, col]) != -1)
                    sensor.AddObservation(1);
                else
                    sensor.AddObservation(0);
            }   
        }*/

        // my target 49
        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                if (settings.TILES[row, col].id == target_tile_id)
                    sensor.AddObservation(1);
                else
                    sensor.AddObservation(0);
            }
        }

        // identifier 4
        sensor.AddOneHotObservation(order, settings.NUM_AGENTS);
    }

    public void FillCommunicationRegions(int agent_order, int[,] communication_regions)
    {
        var rangeRowStart = onTile.row - 2;
        var rangeRowEnd = onTile.row + 2;
        var rangeColStart = onTile.col - 2;
        var rangeColEnd = onTile.col + 2;

        // loop over only -2 to +2 scope
        for (int row = rangeRowStart; row <= rangeRowEnd; row++)
        {
            for (int col = rangeColStart; col <= rangeColEnd; col++)
            {
                if (row < 0 || col < 0 || row >= settings.MAP_HEIGHT || col >= settings.MAP_WIDTH)
                    continue;

                if (settings.AGENTS[row, col] != null)
                {
                    var other_agent_order = settings.AGENTS[row, col].order;
                    communication_regions[agent_order, other_agent_order] = 1;
                }
            }
        }
    }

    public Tile TileAfterAction(int[] act)
    {
        Tile toTile = null;

        if (reached_goal)
        {
            toTile = onTile;
        }
        else
        {
            int action = act[0];
            var floorHelper = new FloorHelper(settings);

            switch (action)
            {
                case Settings.NO_ACTION:
                    toTile = onTile;
                    break;
                case Settings.UP_ACTION:
                    toTile = goUp(floorHelper);
                    break;
                case Settings.DOWN_ACTION:
                    toTile = goDown(floorHelper);
                    break;
                case Settings.RIGHT_ACTION:
                    toTile = goRight(floorHelper);
                    break;
                case Settings.LEFT_ACTION:
                    toTile = goLeft(floorHelper);
                    break;
            }
        }

        if (settings.WALLS[toTile.row, toTile.col] != null)
        {
            return onTile;
        }

        return toTile;
    }


    /// <summary>
    /// Index 0: 0 (stop) or 1 (forward)
    /// Index 1: 0 (none) or 1 (turn right) or 2 (turn left)
    /// Index 2: 0 (none) or 1 (pickup) or 2 (deliver)
    /// </summary>
    /// <param name="actions"></param>
    public void OnActionReceived(int[] actions)
    {
        if (reached_goal)
        {
            //agentHelper.setGlobalReward(-1f / Settings.MAX_ENVIRONMENT_STEPS);
            return;
        }

        MoveAgent(actions);
    }

    public void ResetBetweenSteps()
    {
        CollidedWithWall = false;
        CollidedWithAgent = false;
    }

    public bool WillCollideWithAnotherAgent(int agent_id, int[] agent_action, ActionBuffers all_actions)
    {
        var agent_to_tile = TileAfterAction(agent_action);
        var agent_from = (onTile.row, onTile.col);
        var agent_to = (agent_to_tile.row, agent_to_tile.col);
        
        // to prevent walking queues
        // you can't move to a cell that was occupied when you step into it (even if it will be empty next step)
        if (settings.AGENTS[agent_to_tile.row, agent_to_tile.col] != null && agent_from != agent_to)
        {
            return true;
        }

        var other_agents_from_to = new List<((int, int), (int, int))>();

        foreach (var other_agent_id in Settings.AGENTS_IDs)
        {
            if (agent_id == other_agent_id)
                continue;

            var other_agent_information = settings.AGENT_INFORMATION[other_agent_id];

            var other_agent = other_agent_information.agent;

            var other_agent_action = all_actions.DiscreteActions
                                .Skip(other_agent_information.order * Settings.ACTION_SPACE_PER_AGENT)
                                .Take(Settings.ACTION_SPACE_PER_AGENT).ToArray();

            var other_agent_from_tile = other_agent.onTile;
            var other_agent_to_tile = other_agent.TileAfterAction(other_agent_action);

            other_agents_from_to.Add(((other_agent_from_tile.row, other_agent_from_tile.col), (other_agent_to_tile.row, other_agent_to_tile.col)));
        }


        // ToDO: add && condition to make sure that this duplicate concerns current agent (to generalize to > 2 agents)
        foreach(var other_agent_from_to in other_agents_from_to)
        {
            var other_agent_from = other_agent_from_to.Item1;
            var other_agent_to = other_agent_from_to.Item2;

            if (agent_to == other_agent_to)
                return true;

            // prevent switch behavior
            if (agent_to == other_agent_from && agent_from == other_agent_to)
                return true;
        }

        return false;
    }

    public void MoveAgent(int[] act)
    {
        int action = act[0];
        var floorHelper = new FloorHelper(settings);

        Tile toTile = null;

        switch (action)
        {
            case Settings.NO_ACTION:
                break;
            case Settings.UP_ACTION:
                toTile = goUp(floorHelper);
                break;
            case Settings.DOWN_ACTION:
                toTile = goDown(floorHelper);
                break;
            case Settings.RIGHT_ACTION:
                toTile = goRight(floorHelper);
                break;
            case Settings.LEFT_ACTION:
                toTile = goLeft(floorHelper);
                break;
        }

        // if the agent does moves to another tile (not nothing action)
        if (toTile != null)
        {
            // if this tile contains a wall
            if (settings.WALLS[toTile.row, toTile.col] != null)
            {
                CollidedWithWall = true;
                throw new WallCollision();
            }
            // if this tile is the target
            else if (toTile.agent_id == id)
            {
                transform.position = new Vector3(toTile.posX, transform.position.y, toTile.posZ);
                onTile = toTile;

                _reached_goal = true;
                
                if (agentHelper.AreAllAgentsOnTarget())
                {
                } 
                else
                {
                }
            }
            // otherwise (just move to the tile)
            else
            {
                transform.position = new Vector3(toTile.posX, transform.position.y, toTile.posZ);
                onTile = toTile;
            }
        }
        // if the agent choose to do nothing
        else
        {
        }
    }

    private Tile goUp(FloorHelper floorHelper)
    {
        return floorHelper.up(onTile);
    }

    private Tile goDown(FloorHelper floorHelper)
    {
        return floorHelper.down(onTile);
    }

    private Tile goRight(FloorHelper floorHelper)
    {
        return floorHelper.right(onTile);
    }

    private Tile goLeft(FloorHelper floorHelper)
    {
        return floorHelper.left(onTile);
    }


}
