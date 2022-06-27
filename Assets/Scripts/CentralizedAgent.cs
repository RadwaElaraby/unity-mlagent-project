using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.Linq;
using UnityEngine.UI;

public class CentralizedAgent : Agent
{
    AgentHelper agentHelper;

    Settings settings;

    public GameObject[] ui_wall_collision_counter;
    public GameObject[] ui_agent_collision_counter;
    public GameObject[] ui_goals_reached_counter;

    public void SetSettings(Settings settings)
    {
        this.settings = settings;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var all_actions = GetStoredActionBuffers();

        int[,] communication_regions = new int[settings.NUM_AGENTS, settings.NUM_AGENTS];
        Array.Clear(communication_regions, 0, communication_regions.Length);

        foreach (var id in Settings.AGENTS_IDs)
        {
            //(((9*9)*3 + 4 + 2 ) * 4) + 4*4
            
            var agent_information = settings.AGENT_INFORMATION[id];

            var agent = agent_information.agent;

            // 3*(9*9) + 4
            agent.CollectObservations(sensor);

            agent.FillCommunicationRegions(agent_information.order, communication_regions);

            var agent_action = all_actions.DiscreteActions
                    .Skip(agent_information.order * Settings.ACTION_SPACE_PER_AGENT)
                    .Take(Settings.ACTION_SPACE_PER_AGENT).ToArray();

            // add reward and terminated to observation (won't be used in model) 
            // for the sake of trasmit information between Unity and python
            // 2
            AddCustomObservations(sensor, agent, all_actions, agent_action);

        }

        // 4 * 4
        AddCommunicationObservations(sensor, communication_regions);
    }

    void AddCustomObservations(VectorSensor sensor, MyAgent agent, ActionBuffers all_actions, int[] agent_action)
    {
        // in the step following step where reaching goal happens
        if (agent.reached_goal)
        {
            // in case of finite epsiodes, idle the agent if it has reached its goal
            if (Settings.FINITE_EPISODE)
            {
                sensor.AddObservation(0);
                sensor.AddObservation(1);
                return;
            }
            // otherwise, assign it a new task
            else
            {
                var ui_object = ui_goals_reached_counter[agent.order].GetComponent<Text>();
                ui_object.text = (Convert.ToInt32(ui_object.text) + 1).ToString();

                var floorHelper = new FloorHelper(settings);
                floorHelper.ChangeTargetTileLocation(agent);
                agent._reached_goal = false;
                agent.reached_goal = false;
            }
        }

        // in the step where reaching goal happens
        if (agent._reached_goal)
        {
            agent.reached_goal = true;
            if (Settings.FINITE_EPISODE)
            {
                sensor.AddObservation(1f); //1  // 15
            }
            else
            {
                sensor.AddObservation(1f); /////////////////////// 1 for Game2_10x10_Multi_Dynamic
                                           /////////////////////// 0.1 for Game0_7x7_Multi_Dynamic and Game1_9x9_Multi_Dynamic
                                           /////////////////////// there is no specific reason, I have just found that 0.1 is not sufficient enough!
            }
            sensor.AddObservation(0);
            return;
        }

        
        float reward;
        
        // penalty per timestep
        if (Settings.FINITE_EPISODE)
        {
            reward =  -1f / Settings.MAX_ENVIRONMENT_STEPS; // -1f / Settings.MAX_ENVIRONMENT_STEPS; // -1f;
        }
        else
        {
            reward = 0f;
        }

        // ToDo: generalize to more agents
        if (agent.CollidedWithAgent)
        {
            reward += -3f / Settings.MAX_ENVIRONMENT_STEPS; // -3f / Settings.MAX_ENVIRONMENT_STEPS; // -1f; 
        }
        else if (agent.CollidedWithWall)
        {
            reward += -2f / Settings.MAX_ENVIRONMENT_STEPS; //-2f / Settings.MAX_ENVIRONMENT_STEPS; //  -1f;
        }

        // add the reward to agent observation to overcome problem of shared reward
        sensor.AddObservation(reward);
        // add a terminated signal to agent observation
        sensor.AddObservation(0);
    }

    void AddCommunicationObservations(VectorSensor sensor, int[,] communication_regions)
    {
        for (int row = 0; row < settings.NUM_AGENTS; row++)
        {
            for (int col = 0; col < settings.NUM_AGENTS; col++)
            {
                sensor.AddObservation(communication_regions[row, col]);
            }
        }
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W))
        {
            for (var index = 0; index < settings.NUM_AGENTS; index++ )
                discreteActionsOut[index] = Settings.UP_ACTION;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            for (var index = 0; index < settings.NUM_AGENTS; index++)
                discreteActionsOut[index] = Settings.DOWN_ACTION;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            for (var index = 0; index < settings.NUM_AGENTS; index++)
                discreteActionsOut[index] = Settings.RIGHT_ACTION;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            for (var index = 0; index < settings.NUM_AGENTS; index++)
                discreteActionsOut[index] = Settings.LEFT_ACTION;
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        foreach (var id in Settings.AGENTS_IDs)
        {
            settings.AGENT_INFORMATION[id].agent.ResetBetweenSteps();
        }

        foreach (var id in Settings.AGENTS_IDs)
        {
            var agent_information = settings.AGENT_INFORMATION[id];
            var agent = agent_information.agent;
            var agent_action = actions.DiscreteActions
                                    .Skip(agent_information.order * Settings.ACTION_SPACE_PER_AGENT)
                                    .Take(Settings.ACTION_SPACE_PER_AGENT).ToArray();
            if (agent.WillCollideWithAnotherAgent(id, agent_action, actions))
            {
                agent.CollidedWithAgent = true;
            }
        }

        Array.Clear(settings.AGENTS, 0, settings.AGENTS.Length);

        foreach (var id in Settings.AGENTS_IDs)
        {
            var agent_information = settings.AGENT_INFORMATION[id];

            var individual_actions = actions.DiscreteActions
                                            .Skip(agent_information.order * Settings.ACTION_SPACE_PER_AGENT)
                                            .Take(Settings.ACTION_SPACE_PER_AGENT).ToArray();

            var agent = agent_information.agent;

            if (agent.CollidedWithAgent)
            {
                var ui_object = ui_agent_collision_counter[agent.order].GetComponent<Text>();
                ui_object.text = (Convert.ToInt32(ui_object.text) + 1).ToString();
                StartCoroutine(ChangeColor(0.01f, agent));
                continue;
            }

            try
            {
                agent.OnActionReceived(individual_actions);
            }
            catch (WallCollision e)
            {
                var ui_object = ui_wall_collision_counter[agent.order].GetComponent<Text>();
                ui_object.text = (Convert.ToInt32(ui_object.text) + 1).ToString();
                StartCoroutine(ChangeColor(0.01f, agent));
            }
        }

        foreach (var id in Settings.AGENTS_IDs)
        {
            var agent_information = settings.AGENT_INFORMATION[id];
            var agent = agent_information.agent;

            settings.AGENTS[agent.onTile.row, agent.onTile.col] = agent;


        }
    }

    public IEnumerator ChangeColor(float time, MyAgent agent)
    {
        var renderer = agent.GetComponent<Renderer>();
        var oldColor = renderer.material.color;
        renderer.material.color = Color.white;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        renderer.material.color = oldColor;
    }



}
