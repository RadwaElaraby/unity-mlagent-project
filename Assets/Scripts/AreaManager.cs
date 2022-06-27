using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using UnityEngine.UI;

public class AreaManager : MonoBehaviour
{
    Settings settings;
    FloorHelper floorHelper;
    AgentHelper agentHelper;

    SimpleMultiAgentGroup multiAgentGroup;

    CentralizedAgent centralized_agent;

    public float timeBetweenDecisionsAtInference;
    float m_TimeSinceDecision;

    private int m_ResetTimer;

    public GameObject ui_time_step_counter;

    public Settings getSettings()
    {
        return settings;
    }

    // called only once at the beginning of running the application
    private void Awake()
    {
        settings = GetComponentInChildren<Settings>();
    }


    // called only once at the beginning of running the application
    void Start()
    {
        settings.CreateGameMap();
        BuildArea();
    }

    // called at the beginning of each episode
    public void BuildArea()
    {
        floorHelper = new FloorHelper(settings);
        floorHelper.BuildTiles(transform);
        floorHelper.BuildWalls(transform);

        agentHelper = new AgentHelper(settings);
        agentHelper.InstantiateAgents(transform);

        centralized_agent = GetComponentInChildren<CentralizedAgent>();
        centralized_agent.SetSettings(settings);

        AdjustCameraPosition();
    }

    // called at the beginning of each episode
    public void RebuildArea()
    {
        m_ResetTimer = 0;

        floorHelper = new FloorHelper(settings);
        floorHelper.BuildTiles(transform);
        floorHelper.BuildWalls(transform);

        agentHelper = new AgentHelper(settings);
        agentHelper.ResetAgents(transform);

        ResetUIComponents();
    }


    public void FixedUpdate()
    {
        agentHelper = new AgentHelper(settings);

        // increase timestep counter
        var ui_object = ui_time_step_counter.GetComponent<Text>();
        ui_object.text = (Convert.ToInt32(ui_object.text) + 1).ToString();

        // in case of finite epsiodes, end episode when all agents reaches goal
        if (Settings.FINITE_EPISODE)
        {
            if (agentHelper.AreAllAgentsOnTarget() && m_ResetTimer > 1)
            {
                centralized_agent.EndEpisode();
                settings.CreateRandomGameMap();
                RebuildArea();
            }
        }

        m_ResetTimer += 1;
        if (m_ResetTimer >= Settings.MAX_ENVIRONMENT_STEPS && Settings.MAX_ENVIRONMENT_STEPS > 0)
        {
            centralized_agent.EpisodeInterrupted();
            settings.CreateRandomGameMap();
            RebuildArea();
        }

        WaitTimeInference();
    }

    void WaitTimeInference()
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            centralized_agent.RequestDecision();
        }
        else
        {
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                m_TimeSinceDecision = 0f;
                centralized_agent.RequestDecision();
            }
            else
            {
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }

    public void ResetUIComponents()
    {
        foreach (var ui in centralized_agent.ui_goals_reached_counter)
        {
            ui.GetComponent<Text>().text = "0";
        }
        foreach (var ui in centralized_agent.ui_agent_collision_counter)
        {
            ui.GetComponent<Text>().text = "0";
        }
        foreach (var ui in centralized_agent.ui_wall_collision_counter)
        {
            ui.GetComponent<Text>().text = "0";
        }

        ui_time_step_counter.GetComponent<Text>().text = "0";
    }

    public void AdjustCameraPosition()
    {
        GetComponentInChildren<Camera>().transform.position = Game2_10x10_Dynamic_Walls.CAMERA_POSITION;
    }

}
