package pl.wsb.fitnesstracker.training.api;

import pl.wsb.fitnesstracker.training.api.TrainingDto;

import java.util.List;
import java.util.Optional;

/**
 * Contract for providing training data to API layer.
 */
public interface TrainingProvider {

    /**
     * Retrieves a training by its ID.
     *
     * @param trainingId training identifier
     * @return Optional training DTO
     */
    Optional<TrainingDto> getTraining(Long trainingId);

    /**
     * Retrieves all trainings.
     *
     * @return list of all training DTOs
     */
    List<TrainingDto> getAllTrainings();

    /**
     * Retrieves trainings assigned to a given user.
     *
     * @param userId user identifier
     * @return list of user trainings as DTOs
     */
    List<TrainingDto> getTrainingsByUserId(Long userId);
}
