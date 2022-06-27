using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class RLAgent : Agent
{
    Rigidbody rigidBody;

    public float forceSpeed = 1000f;
    public float rotateSpeed = 300f;

    private Area area;

    bool setupExecuted = false;

    public float initialPosX;
    public float initialPosY;
    public float initialPosZ;

    public float initialRotX;
    public float initialRotY;
    public float initialRotZ;
    public float initialRotW;

    float scaleX;
    float scaleZ;

    float minPosX;
    float maxPosX;

    float minPosZ;
    float maxPosZ;

    float minRelPosX;
    float maxRelPosX;

    float minRelPosZ;
    float maxRelPosZ;

    Floor floor;
    [HideInInspector]
    public BoxCollider floorCollider;

    public override void Initialize()
    {
        if (!setupExecuted)
            Setup();
    }

    void Setup()
    {
        rigidBody = GetComponent<Rigidbody>();
        area = GetComponentInParent<Area>();
        floor = area.GetComponentInChildren<Floor>();
        floorCollider = floor.GetComponent<BoxCollider>();

        setupExecuted = true;

        initialPosX = transform.position.x;
        initialPosY = transform.position.y;
        initialPosZ = transform.position.z;

        initialRotX = transform.rotation.x;
        initialRotY = transform.rotation.y;
        initialRotZ = transform.rotation.z;
        initialRotW = transform.rotation.w;

        scaleX = transform.localScale.x;
        scaleZ = transform.localScale.z;

        minPosX = floorCollider.bounds.min.x + scaleX/2;
        maxPosX = floorCollider.bounds.max.x + -scaleX/2;

        minPosZ = floorCollider.bounds.min.z + scaleZ/2;
        maxPosZ = floorCollider.bounds.max.z + -scaleZ/2; 
        
        minRelPosX = -floorCollider.bounds.size.x;
        maxRelPosX = floorCollider.bounds.size.x;

        minRelPosZ = -floorCollider.bounds.size.z;
        maxRelPosZ = floorCollider.bounds.size.z;
    }

    private void FixedUpdate()
    {
        var y = this.tag;
        var x = StepCount;

    }
    private void Update()
    {
        var x = StepCount;

    }

    public override void OnEpisodeBegin()
    {
        if (!setupExecuted)
            Setup();

        transform.position = new Vector3(initialPosX, initialPosY, initialPosZ);
        transform.rotation = new Quaternion(initialRotX, initialRotY, initialRotZ, initialRotW);
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
    }

    /// <summary>
    /// Index 0: 0 (stop) or 1 (forward)
    /// Index 1: 0 (none) or 1 (turn right) or 2 (turn left)
    /// Index 2: 0 (none) or 1 (pickup) or 2 (deliver)
    /// </summary>
    /// <param name="actions"></param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveAgent(actions.DiscreteActions);

        AddReward(-1 / MaxStep);

    }

    /// <summary>
    /// Index 0: 0 (stop) or 1 (forward)
    /// Index 1: 0 (none) or 1 (turn right) or 2 (turn left)
    /// Index 2: 0 (none) or 1 (pickup) or 2 (deliver)
    /// </summary>
    /// <param name="actions"></param>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * rotateSpeed);
        rigidBody.AddForce(dirToGo * forceSpeed, ForceMode.VelocityChange);
    }

    public void ScoredAGoal()
    {
        AddReward(5f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);
    }

    /*
    public override void CollectObservations(VectorSensor sensor)
    {
        var agentPosXNormalized = (transform.position.x - minPosX) / (maxPosX - minPosX);
        var agentPosZNormalized = (transform.position.z - minPosZ) / (maxPosZ - minPosZ);
        sensor.AddObservation(agentPosXNormalized);
        sensor.AddObservation(agentPosZNormalized);

        var agentRotNormalized = transform.rotation.eulerAngles / 360.0f;
        sensor.AddObservation(agentRotNormalized.y);

        var nearestItem = area.FindNearestItem();
        if (nearestItem)
        {
            var nearestItemRelPos = nearestItem.transform.position - transform.position;
            var nearestItemRelPosXNormalized = (nearestItemRelPos.x - minRelPosX) / (maxRelPosX - minRelPosX);
            var nearestItemRelPosZNormalized = (nearestItemRelPos.z - minRelPosZ) / (maxRelPosZ - minRelPosZ);

            sensor.AddObservation(nearestItemRelPosXNormalized);
            sensor.AddObservation(nearestItemRelPosZNormalized);

            var targetRelPos = nearestItem.target.transform.position - transform.position;
            var targetRelPosXNormalized = (targetRelPos.x - minRelPosX) / (maxRelPosX - minRelPosX);
            var targetRelPosZNormalized = (targetRelPos.z - minRelPosZ) / (maxRelPosZ - minRelPosZ);

            sensor.AddObservation(targetRelPosXNormalized);
            sensor.AddObservation(targetRelPosZNormalized);
        }
        else
        {
            sensor.AddObservation(0);
            sensor.AddObservation(0);

            sensor.AddObservation(0);
            sensor.AddObservation(0);
        }
        
        
        //var item = PickedItem ? PickedItem : Area.FindNearestItem(this);
        //sensor.AddObservation(item.transform.localPosition.normalized);
        //sensor.AddObservation((item.transform.localPosition - gameObject.transform.localPosition).normalized);

        //sensor.AddObservation(Target.transform.localPosition.normalized);
        //sensor.AddObservation((Target.transform.localPosition - gameObject.transform.localPosition).normalized);

    }*/

}
