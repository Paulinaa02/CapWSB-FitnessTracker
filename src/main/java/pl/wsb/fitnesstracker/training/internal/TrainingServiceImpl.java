package pl.wsb.fitnesstracker.training.internal;

import org.springframework.stereotype.Service;
import pl.wsb.fitnesstracker.training.api.TrainingProvider;
import pl.wsb.fitnesstracker.training.api.TrainingRepository;
import pl.wsb.fitnesstracker.training.api.TrainingDto;

import java.util.List;
import java.util.Optional;

/**
 * Service implementation for training read operations.
 */
@Service
public class TrainingServiceImpl implements TrainingProvider {

    private final TrainingRepository trainingRepository;
    private final TrainingMapper trainingMapper;

    public TrainingServiceImpl(final TrainingRepository trainingRepository,
                               final TrainingMapper trainingMapper) {
        this.trainingRepository = trainingRepository;
        this.trainingMapper = trainingMapper;
    }

    @Override
    public Optional<TrainingDto> getTraining(final Long trainingId) {
        return trainingRepository.findById(trainingId)
                .map(trainingMapper::toDto);
    }

    @Override
    public List<TrainingDto> getAllTrainings() {
        return trainingRepository.findAll()
                .stream()
                .map(trainingMapper::toDto)
                .toList();
    }

    @Override
    public List<TrainingDto> getTrainingsByUserId(final Long userId) {
        return trainingRepository.findAllByUser_Id(userId)
                .stream()
                .map(trainingMapper::toDto)
                .toList();
    }
}
