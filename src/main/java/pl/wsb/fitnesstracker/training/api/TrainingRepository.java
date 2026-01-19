package pl.wsb.fitnesstracker.training.api;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TrainingRepository extends JpaRepository<Training, Long> {

    /**
     * Retrieves trainings for a specific user.
     *
     * @param userId user identifier
     * @return list of trainings for user
     */
    List<Training> findAllByUser_Id(Long userId);
}
